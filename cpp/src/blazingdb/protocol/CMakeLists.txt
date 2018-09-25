set(SRCS api.cpp)

add_library(blazingdb-protocol ${SRCS})

install(FILES api.h DESTINATION include/blazingdb/protocol)

install(TARGETS blazingdb-protocol
  ARCHIVE
    DESTINATION lib
    COMPONENT API
  LIBRARY
    DESTINATION lib
    COMPONENT API
  PUBLIC_HEADER
    DESTINATION include
    COMPONENT API)
