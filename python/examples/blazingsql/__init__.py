from cudf import _gdf
from cudf.dataframe.column import Column
from cudf import DataFrame
from cudf.dataframe.dataframe import Series
from cudf.dataframe.buffer import Buffer
from cudf import utils

import numpy as np
import pandas as pd

import contextlib
from blazingdb.protocol.transport.channel import ResponseErrorSchema
from blazingdb.protocol.transport.channel import MakeAuthRequestBuffer
from blazingdb.protocol.errors import Error
from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.orchestrator import OrchestratorMessageType
from blazingdb.protocol.transport.channel import ResponseSchema
from blazingdb.protocol import UnixSocketConnection
from blazingdb.protocol import Client
from blazingdb.protocol.orchestrator import AuthResponseSchema
from blazingdb.protocol.orchestrator import DDLCreateTableRequestSchema
from blazingdb.protocol.transport.channel import MakeRequestBuffer

from blazingdb.protocol.orchestrator import DDLDropTableRequestSchema
from blazingdb.protocol.orchestrator import BuildDMLRequestSchema
from blazingdb.protocol.orchestrator import DMLResponseSchema


from blazingdb.protocol.interpreter import GetResultRequestSchema
from blazingdb.protocol.interpreter import GetQueryResultFrom

from numba import cuda
from numba.cuda.cudadrv import driver, devices

require_context = devices.require_context
current_context = devices.get_context
gpus = devices.gpus

@contextlib.contextmanager
def open_connection(path):
    connection = Connection(path)
    yield connection
    connection.close()


class Connection:
    """
     Do IPC calls between clients and artifacts.
     """
    def __init__(self, path):
        self._path = path
        self._unix_connection = UnixSocketConnection(self._path)
        self._socket = Client(self._unix_connection)
        self._databases = []
        self.accessToken = None
        self._open()

    def _open(self):
        requestBuffer = MakeAuthRequestBuffer(OrchestratorMessageType.AuthOpen)
        try:
            responseBuffer = self.send(requestBuffer)
        except ConnectionError as err:
            print(err)
        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)
        else:
            self.accessToken = AuthResponseSchema.From(response.payload).accessToken
        return self.accessToken

    def send(self, buffer):
        return self._socket.send(buffer)

    def close(self):
        requestBuffer = MakeAuthRequestBuffer(OrchestratorMessageType.AuthClose)
        try:
            responseBuffer = self.send(requestBuffer)
        except ConnectionError as err:
            print(err)

        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)

        self._socket = None
        self._unix_connection = None
        self.accessToken = None

    def Database(self, name='main'):
        """
           Returns database object.
           """
        db = Database(name, self, lambda: self._databases.remove(db) if db in self._databases else None)
        self._databases.append(db)
        return db


class Database:

    def __init__(self, name, connection, cleaner):
        self._clean = cleaner
        self._name = name
        self._connection = connection
        self._tables = []

    def __del__(self):
        """Drop tables."""
        self.drop()

    def Table(self, name, data_frame):
        """
        Argsuments:
          dataframe: data_frame dataframe
        Returns table object.
        """
        table = Table(name, data_frame,
                      lambda: self._tables.remove(table) if table in self._tables else None)
        self._tables.append(table)

        column_names = list(data_frame.columns)
        def gdf_dtype_to_str(gdf_dtype):
            return {
                libgdf.GDF_FLOAT64: 'GDF_FLOAT64',
                libgdf.GDF_FLOAT32: 'GDF_FLOAT32',
                libgdf.GDF_INT64: 'GDF_INT64',
                libgdf.GDF_INT32: 'GDF_INT32',
                libgdf.GDF_INT16: 'GDF_INT16',
                libgdf.GDF_INT8: 'GDF_INT8',
                libgdf.GDF_DATE64: 'GDF_DATE64',
                libgdf.N_GDF_TYPES: 'N_GDF_TYPES',
                libgdf.GDF_CATEGORY: 'GDF_CATEGORY',
            }[gdf_dtype]
        column_types = [gdf_dtype_to_str(_gdf.np_to_gdf_dtype(dtype))  for dtype in list(data_frame.dtypes)]
        self._create_table(name, column_names, column_types, self._name)
        return table

    def _create_table(self, tableName, columnNames, columnTypes, dbName):
        create_table_message = DDLCreateTableRequestSchema(name=tableName,
                                                           columnNames=columnNames,
                                                           columnTypes=columnTypes,
                                                           dbName=dbName)
        requestBuffer = MakeRequestBuffer(OrchestratorMessageType.DDL_CREATE_TABLE,
                                          self._connection.accessToken,
                                          create_table_message)
        responseBuffer = self._connection.send(requestBuffer)
        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)
        return response.status

    def remove_table(self, table_name):
        """
        Args:
          table: table name.
        """
        drop_table_message = DDLDropTableRequestSchema(name=table_name, dbName=self._name)
        requestBuffer = MakeRequestBuffer(OrchestratorMessageType.DDL_DROP_TABLE,
                                          self._connection.accessToken,
                                          drop_table_message)
        responseBuffer = self._connection.sendRequest(requestBuffer)
        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)
        return response.status

    def run_query(self, query, tables):
        """
        Returns token object.
        """
        tableGroup = self._get_table_group(tables)

        dml_message = BuildDMLRequestSchema(query, tableGroup)
        requestBuffer = MakeRequestBuffer(OrchestratorMessageType.DML,
                                          self._connection.accessToken,
                                          dml_message)
        responseBuffer = self._connection.send(requestBuffer)
        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)

        result = DMLResponseSchema.From(response.payload)

        return (result.resultToken, result.nodeConnection.path)


    def _send_request(self, unix_path, requestBuffer):
        connection = UnixSocketConnection(unix_path)
        client = Client(connection)
        return client.send(requestBuffer)


    @contextlib.contextmanager
    def get_result(self, result_token, connection_info):
        getResultRequest = GetResultRequestSchema(resultToken=result_token)

        requestBuffer = MakeRequestBuffer(InterpreterMessage.GetResult,
                                          self._connection.accessToken,
                                          getResultRequest)

        responseBuffer = self._send_request(connection_info, requestBuffer)

        response = ResponseSchema.From(responseBuffer)

        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)

        resultSet = GetQueryResultFrom(response.payload)

        def cffi_view_to_column_mem(cffi_view):
            data = _gdf._as_numba_devarray(intaddr=int(ffi.cast("uintptr_t",
                                                                cffi_view.data)),
                                           nelem=cffi_view.size,
                                           dtype=_gdf.gdf_to_np_dtype(cffi_view.dtype))
            mask = None
            return data, mask

        def from_cffi_view(cffi_view):
            data_mem, mask_mem = cffi_view_to_column_mem(cffi_view)
            data_buf = Buffer(data_mem)
            mask = None
            return column.Column(data=data_buf, mask=mask)

        gdf_columns = []
        ipchandles = []
        for i, c in enumerate(resultSet.columns):
            assert len(c.data) == 64
            ipch, data_ptr = self._open_ipc_array(c.data, shape=c.size, dtype=_gdf.gdf_to_np_dtype(c.dtype))
            ipchandles.append(ipch)
            gdf_col = _gdf.columnview_from_devary(data_ptr, ffi.NULL)
            newcol = from_cffi_view(gdf_col)
            gdf_columns.append(newcol.view(NumericalColumn, dtype=newcol.dtype))

        df = DataFrame()
        for k, v in zip(resultSet.columnNames, gdf_columns):
            df[str(k)] = v
        yield df

        for ipch in ipchandles:
            ipch.close()

    def _open_ipc_array(self, handle, shape, dtype, strides=None, offset=0):
        dtype = np.dtype(dtype)
        # compute size
        size = np.prod(shape) * dtype.itemsize
        # manually recreate the IPC mem handle
        handle = driver.drvapi.cu_ipc_mem_handle(*handle)
        # use *IpcHandle* to open the IPC memory
        ipchandle = driver.IpcHandle(None, handle, size, offset=offset)
        return ipchandle, ipchandle.open_array(current_context(), shape=shape,
                                               strides=strides, dtype=dtype)


    def drop(self):
        for table in self._tables:
            table.drop()
        self._clean()

    def _get_table_group(self, tables):
        tableGroup = {}
        tableGroup["name"] = ""
        tableGroup["tables"] = []
        for inputData in tables:
            table = {}
            table["name"] = inputData.name
            table["columns"] = []
            table["columnNames"] = []
            for name, series in inputData.data_frame._cols.items():
                table["columnNames"].append(name)
                cffiView = series._column.cffi_view
                if series._column._mask is None:
                    table["columns"].append(
                        {'data': bytes(series._column._data.mem.get_ipc_handle()._ipc_handle.handle),
                         'valid': b"",
                         'size': cffiView.size,
                         'dtype': cffiView.dtype,
                         'dtype_info': 0,  # TODO dtype_info is currently not used in cudf
                         'null_count': cffiView.null_count})
                else:
                    table["columns"].append(
                        {'data': bytes(series._column._data.mem.get_ipc_handle()._ipc_handle.handle),
                         'valid': bytes(series._column._mask.mem.get_ipc_handle()._ipc_handle.handle),
                         'size': cffiView.size,
                         'dtype': cffiView.dtype,
                         'dtype_info': 0,  # TODO dtype_info is currently not used in cudf
                         'null_count': cffiView.null_count})
            tableGroup["tables"].append(table)
        return tableGroup


class Table:
    """Table object."""

    def __init__(self, name, data_frame, cleaner):
        self._clean = cleaner
        self.name = name
        self.data_frame = data_frame

    def __del__(self):
        self.drop()

    def drop(self):
        self._clean()


class ConnectionError():
  """"""


class SintaxError():
  """"""


class TokenNotFoundError():
  """"""