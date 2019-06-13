#ifndef BLAZINGDB_PROTOCOL_CALCITE_ERRORS_H_
#define BLAZINGDB_PROTOCOL_CALCITE_ERRORS_H_

#include <blazingdb/protocol/message/errors.h>

namespace blazingdb {
namespace protocol {
namespace calcite {
namespace errors {

using protocol::errors::Error;

//! Thrown by Calcite calls when the query-string is invalid.

class SyntaxError : public Error {};

}  // namespace errors
}  // namespace calcite
}  // namespace protocol
}  // namespace blazingdb

#endif
