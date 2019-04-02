import os
import logging

logging.debug('debug')
logging.info('info')
logging.warning('warning')
logging.error('error')
logging.critical('critical')

##############################################################################
log_format = (
    '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s')

logging.basicConfig(
    filename='debug.log',
    format=log_format,
    level=logging.DEBUG,
)

formatter = logging.Formatter(log_format)
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
##############################################################################

from blazingdb.protocol.transport.channel import ResponseErrorSchema
from blazingdb.protocol.transport.channel import MakeAuthRequestBuffer
from blazingdb.protocol.errors import Error
from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.protocol.orchestrator import OrchestratorMessageType
from blazingdb.protocol.transport.channel import ResponseSchema
from blazingdb.protocol import UnixSocketConnection
from blazingdb.protocol import Client
from blazingdb.protocol.orchestrator import AuthResponseSchema


class Authentication:
    """
     Do IPC calls between clients and artifacts.
     """
    def __init__(self, path):
        self._path = path
        self._unix_connection = UnixSocketConnection(self._path)
        self._socket = Client(self._unix_connection)
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
        # TODO find a way to print only for debug mode (add verbose arg)
        return self.accessToken

    def send(self, buffer):
        return self._socket.send(buffer)

    def __del__(self):
        self.close()

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

    def Connector(self, name='main'):
        """
           Returns database object.
           """
        db = Connector(name, self)
        return db



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


class Connector:

    def __init__(self, name, connection):
        self.dbname = name
        self.connection = connection

    def __del__(self):
        """Drop tables."""
        self.drop()

    def _get_table_def_from_gdf(gdf):
        cols = gdf.columns.values.tolist()

        # TODO find a way to print only for debug mode (add verbose arg)
        # print(cols)

        types = []
        for key, column in gdf._cols.items():
            dtype = column._column.cffi_view.dtype
            types.append(gdf_column_type_to_str(dtype))
        return cols, types

    def _reset_table(self, table, gdf):
        self.remove_table(table)
        cols, types = self._get_table_def_from_gdf(gdf)
        self._create_table(table, cols, types)

    def Table(self, name, gdf):
        """
        Argsuments:
          dataframe: gdf dataframe
        Returns table object.
        """
        table = Table(name, gdf)

        return table

    def _create_table(self, tableName, columnNames, columnTypes):
        create_table_message = DDLCreateTableRequestSchema(name=tableName,
                                                           columnNames=columnNames,
                                                           columnTypes=columnTypes,
                                                           dbName=self.dbname)
        requestBuffer = MakeRequestBuffer(OrchestratorMessageType.DDL_CREATE_TABLE,
                                          self.connection.accessToken,
                                          create_table_message)
        responseBuffer = self.connection.send(requestBuffer)
        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)
        return response.status

    def remove_table(self, table_name):
        """
        Args:
          table: table name.
        """
        drop_table_message = DDLDropTableRequestSchema(name=table_name, dbName=self.dbname)
        requestBuffer = MakeRequestBuffer(OrchestratorMessageType.DDL_DROP_TABLE,
                                          self.connection.accessToken,
                                          drop_table_message)
        responseBuffer = self.connection.sendRequest(requestBuffer)
        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)
        return response.status

    def run_query(self, query, tables):
        """
        Returns token object.
        """
        tableGroup = self._to_table_group(tables)

        dml_message = BuildDMLRequestSchema(query, tableGroup)
        requestBuffer = MakeRequestBuffer(OrchestratorMessageType.DML,
                                          self.connection.accessToken,
                                          dml_message)
        responseBuffer = self.connection.send(requestBuffer)
        response = ResponseSchema.From(responseBuffer)
        if response.status == Status.Error:
            errorResponse = ResponseErrorSchema.From(response.payload)
            if b'SqlSyntaxException' in errorResponse.errors:
                raise SyntaxError(errorResponse.errors.decode('utf-8'))
            raise Error(errorResponse.errors)
        result = DMLResponseSchema.From(response.payload)
        return (result.resultToken, result.nodeConnection.path, result.calciteTime)


    def _send_request(self, unix_path, requestBuffer):
        connection = UnixSocketConnection(unix_path)
        client = Client(connection)
        return client.send(requestBuffer)


    def get_result(self, result_token, interpreter_path):
        getResultRequest = GetResultRequestSchema(resultToken=result_token)

        requestBuffer = MakeRequestBuffer(InterpreterMessage.GetResult,
                                          self.connection.accessToken,
                                          getResultRequest)

        responseBuffer = self._send_request(interpreter_path, requestBuffer)

        response = ResponseSchema.From(responseBuffer)

        if response.status == Status.Error:
            raise Error(ResponseErrorSchema.From(response.payload).errors)

        queryResult = GetQueryResultFrom(response.payload)
        return queryResult


    def _to_table_group(self, tables):
        database_name = self.dbname
        tableGroup = {'name': database_name}
        blazing_tables = []
        for table, gdf in tables.items():
            # TODO columnNames should have the columns of the query (check this)
            blazing_table = {'name': database_name + '.' + table,
                             'columnNames': gdf.columns.values.tolist()}
            blazing_columns = []

            for column in gdf.columns:
                dataframe_column = gdf._cols[column]
                # TODO support more column types
                numerical_column = dataframe_column._column
                data_sz = numerical_column.cffi_view.size
                dtype = numerical_column.cffi_view.dtype

                data_ipch = get_ipc_handle_for(dataframe_column)

                # TODO this valid data is fixed and is invalid
                # felipe doesnt undertand why we need this we can send null
                # if the bitmask is not valid
                # sample_valid_df = gen_data_frame(data_sz, 'valid', np.int8)
                # valid_ipch = get_ipc_handle_for(sample_valid_df['valid'])

                blazing_column = {
                    'data': data_ipch,
                    'valid': None,  # TODO we should use valid mask
                    'size': data_sz,
                    'dtype': dataframe_column._column.cffi_view.dtype,
                    'null_count': 0,
                    'dtype_info': 0
                }
                blazing_columns.append(blazing_column)

            blazing_table['columns'] = blazing_columns
            blazing_tables.append(blazing_table)

        tableGroup['tables'] = blazing_tables
        return tableGroup


class BlazingSqlIntance:
    _singleton = None

    def __new__(cls):
        if not cls._singleton:
            connection = AuthenticationIntance()
            cls._singleton = connection.Connector('main')
            logging.info('Init BlazingSqlInstance')

        return cls._singleton


class AuthenticationIntance:
    _singleton = None

    def __new__(cls):
        if not cls._singleton:
            cls._singleton = Authentication('/tmp/orchestrator.socket')
            logging.info('Init BlazingSqlInstance')
        return cls._singleton



from libgdf_cffi import ffi
from cudf.dataframe.datetime import DatetimeColumn
from cudf.dataframe.numerical import NumericalColumn

import pyarrow as pa
from cudf import _gdf
from cudf.dataframe.column import Column
from cudf import DataFrame
from cudf.dataframe.dataframe import Series
from cudf.dataframe.buffer import Buffer
from cudf import utils

from numba import cuda
import numpy as np
import pandas as pd

import time
def _private_run_query(sql, tables):
    startTime = time.time()

    client = BlazingSqlIntance()

    try:
        for table, gdf in tables.items():
            client._reset_table(table, gdf)
    except Error as err:
        print(err)
    ipchandles = []
    resultSet = None
    token = None
    interpreter_path = None
    try:
        tableGroup = _to_table_group(tables)
        token, interpreter_path, calciteTime = client.run_dml_query_token(sql, tableGroup)
        resultSet = client._get_result(token, interpreter_path)

        # TODO: this function was copied from column.py in cudf  but fixed so that it can handle a null mask. cudf has a bug there
        def from_cffi_view(cffi_view):
            """Create a Column object from a cffi struct gdf_column*.
            """
            data_mem, mask_mem = _gdf.cffi_view_to_column_mem(cffi_view)
            data_buf = Buffer(data_mem)

            if mask_mem is not None:
                mask = Buffer(mask_mem)
            else:
                mask = None

            return Column(data=data_buf, mask=mask)

        # TODO: this code does not seem to handle nulls at all. This will need to be addressed
        def _open_ipc_array(handle, shape, dtype, strides=None, offset=0):
            dtype = np.dtype(dtype)
            # compute size
            size = np.prod(shape) * dtype.itemsize
            # manually recreate the IPC mem handle
            handle = driver.drvapi.cu_ipc_mem_handle(*handle)
            # use *IpcHandle* to open the IPC memory
            ipchandle = driver.IpcHandle(None, handle, size, offset=offset)
            return ipchandle, ipchandle.open_array(current_context(), shape=shape,
                                                   strides=strides, dtype=dtype)

        gdf_columns = []

        for i, c in enumerate(resultSet.columns):
            assert len(c.data) == 64
            ipch, data_ptr = _open_ipc_array(
                c.data, shape=c.size, dtype=_gdf.gdf_to_np_dtype(c.dtype))
            ipchandles.append(ipch)

            # TODO: this code imitates what is in io.py from cudf in read_csv . The way it handles datetime indicates that we will need to fix this for better handling of timestemp and other datetime data types
            cffi_view = _gdf.columnview_from_devary(data_ptr, ffi.NULL)
            newcol = from_cffi_view(cffi_view)
            if (newcol.dtype == np.dtype('datetime64[ms]')):
                gdf_columns.append(newcol.view(DatetimeColumn, dtype='datetime64[ms]'))
            else:
                gdf_columns.append(newcol.view(NumericalColumn, dtype=newcol.dtype))

        df = DataFrame()
        for k, v in zip(resultSet.columnNames, gdf_columns):
            df[str(k)] = v

        resultSet.columns = df

        totalTime = (time.time() - startTime) * 1000  # in milliseconds

        # @todo close ipch, see one solution at ()
        # print(df)
        # for ipch in ipchandles:
        #     ipch.close()
    except SyntaxError as error:
        raise error
    except Error as err:
        print(err)

    return_result = ResultSetHandle(resultSet.columns, token, interpreter_path, ipchandles, client, calciteTime,
                                    resultSet.metadata.time, totalTime)
    return return_result


class ResultSetHandle:

    columns = None
    token = None
    interpreter_path = None
    handle = None
    client = None

    def __init__(self,columns, token, interpreter_path, handle, client, calciteTime, ralTime, totalTime):
        self.columns = columns
        self.token = token
        self.interpreter_path = interpreter_path
        self.handle = handle
        self.client = client
        self.calciteTime = calciteTime
        self.ralTime = ralTime
        self.totalTime = totalTime

    def __del__(self):
        del self.handle
        self.client.free_result(self.token,self.interpreter_path)

    def __str__(self):
      return ('''columns = %(columns)s
token = %(token)s
interpreter_path = %(interpreter_path)s
handle = %(handle)s
client = %(client)s
calciteTime = %(calciteTime)d
ralTime = %(ralTime)d
totalTime = %(totalTime)d''' % {
        'columns': self.columns,
        'token': self.token,
        'interpreter_path': self.interpreter_path,
        'handle': self.handle,
        'client': self.client,
        'calciteTime' : self.calciteTime,
        'ralTime' : self.ralTime,
        'totalTime' : self.totalTime,
      })

    def __repr__(self):
      return str(self)


class Table:
    """Table object."""

    def __init__(self, name, gdf, cleaner):
        self._clean = cleaner
        self.name = name
        self.gdf = gdf

    def __del__(self):
        self.drop()

    def drop(self):
        self._clean()


class SchemaFrom:
    Gdf = 0
    ParquetFile = 1
    CsvFile = 2


class Schema:

    def __init__(self, type, **kwargs):
        self.schema_type = type

        schema_type, names, types = self._get_schema(**kwargs)
        assert schema_type == self.schema_type

        self.names = names
        self.types = types

        # client = _get_client()
        # client.run_ddl_drop_table(table_name, 'main')
        # column_types = [dtype_to_str(item)  for item in list(dtype)]
        # print(column_names)
        # print(column_types)
        # client.run_ddl_create_table(table_name, column_names, column_types, 'main')

    def __hash__(self):
        return hash(self.schema_type)

    def __eq__(self, other):
        return self.schema_type == other.schema_type

    def _get_schema_from_gdf(self, gdf_table):
        return None

    def _get_schema_from_parquet(self, path):
        return None

    def _get_schema(self, **kwargs):
        """
        :param table_name:
        :param kwargs:
                csv: column_names, column_types
                gdf: gpu data frame
                parquet: path
        :return:
        """
        column_names = kwargs.get('column_names', None)
        column_types = kwargs.get('column_types', None)
        gdf = kwargs.get('gdf', None)
        path = kwargs.get('path', None)

        if column_names is not None and column_types is not None:
            return SchemaFrom.CsvFile, column_names, column_types
        elif gdf is not None:
            return SchemaFrom.CsvFile, self._get_schema_from_gdf(gdf)
        elif path is not None:
            return SchemaFrom.ParquetFile, self._get_schema_from_parquet(path)

        schema_logger = logging.getLogger('Schema')
        schema_logger.critical('Not schema found')


def gdf_column_type_to_str(dtype):
    str_dtype = {
        0: 'GDF_invalid',
        1: 'GDF_INT8',
        2: 'GDF_INT16',
        3: 'GDF_INT32',
        4: 'GDF_INT64',
        5: 'GDF_FLOAT32',
        6: 'GDF_FLOAT64',
        7: 'GDF_DATE32',
        8: 'GDF_DATE64',
        9: 'GDF_TIMESTAMP',
        10: 'GDF_CATEGORY',
        11: 'GDF_STRING',
        12: 'GDF_UINT8',
        13: 'GDF_UINT16',
        14: 'GDF_UINT32',
        15: 'GDF_UINT64',
        16: 'N_GDF_TYPES'
    }
    return str_dtype[dtype]