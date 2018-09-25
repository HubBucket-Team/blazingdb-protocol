set(SRCS api.cpp)

add_library(blazingdb-protocol ${SRCS})

set_target_properties(blazingdb-protocol
  PROPERTIES
    PUBLIC_HEADER "api.h")

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
