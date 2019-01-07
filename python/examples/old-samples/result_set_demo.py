from blazingdb.messages.blazingdb.protocol.interpreter import \
  BlazingMetadata, GetResultResponse

import blazingdb.protocol as bp
import blazingdb.protocol.gdf as bpg
import blazingdb.protocol.orchestrator as bpo
import blazingdb.protocol.interpreter as bpi
from pprint import pprint

def createcol():
  data = bpg.cudaIpcMemHandle_tSchema(reserved=b'data')
  valid = bpg.cudaIpcMemHandle_tSchema(reserved=b'valid')
  dtype_info = bpg.gdf_dtype_extra_infoSchema(time_unit=0)

  return bpg.gdf_columnSchema(data=data, valid=valid, size=10,
    dtype=0, dtype_info=dtype_info, null_count=0)

col1 = createcol()
col2 = createcol()


bme = bpi.BlazingMetadataSchema(
  status=b'STAT',
  message=b'MESG',
  time=100,
  rows=122,
)

tableA = bpi.GetResultResponseSchema(
  metadata=bme,
  columns=[col1, col2],
  columnNames=['id', 'age']
)

b = tableA.ToBuffer()

dto = bpi.GetResultResponseSchema.From(b)

pprint(list(c.data.reserved for c in dto.columns))