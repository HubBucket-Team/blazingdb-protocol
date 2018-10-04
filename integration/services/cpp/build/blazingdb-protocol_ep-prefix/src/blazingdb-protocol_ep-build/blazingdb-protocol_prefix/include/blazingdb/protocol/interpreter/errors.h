#ifndef BLAZINGDB_PROTOCOL_INTERPRETER_ERRORS_H_
#define BLAZINGDB_PROTOCOL_INTERPRETER_ERRORS_H_

#include <blazingdb/protocol/errors.h>

namespace blazingdb {
namespace protocol {
namespace interpreter {
namespace errors {

using protocol::errors::Error;

/*!
 * Thrown by Interpreter calls when it is passed a plan
 * with invalid statements.
 */

class LogicalPlanError : public Error {};

/*!
 * Thrown by Interpreter.RunSql(query) calls when it can not to commit
 * a result token for the query.
 */
class TokenCreationError : public Error {};

/*!
 * Thrown by Interpreter.GetResult(token) calls when there are not results
 * for the requested token.
 */

class TokenNotFoundError : public Error {};

}  // namespace errors
}  // namespace interpreter
}  // namespace protocol
}  // namespace blazingdb

#endif
