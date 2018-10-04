//! Errors used in the BlazingDB Protocol API.

#ifndef BLAZINGDB_PROTOCOL_ERRORS_H_
#define BLAZINGDB_PROTOCOL_ERRORS_H_

#include <exception>

namespace blazingdb {
namespace protocol {
namespace errors {

//! Base protocol error type.

class Error : public std::exception {};

//! An internal server error. Please report this to BlazingDB.

class InternalError : public Error {};

/*!
 * Thrown by client calls to a unavailable server.
 * This can happen when you attempt to create a client or send a buffer,
 * or if the server is overloaded o having trouble.
 */

class CommunicationError : public Error {};

//! A communication error on IPC context.

class IPCError : public CommunicationError {};

}  // namespace errors
}  // namespace protocol
}  // namespace blazingdb

#endif
