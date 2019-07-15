#include "connection.h"

namespace blazingdb {
namespace protocol {

int ConnectionUtils::parsePort(const char* port_str) {
    char *t;
    const long int port = strtol(port_str, &t, 10);
    if (*t != '\0') {
        return -1;
    } else {
        if (port > 0) {
            return port;
        } else {
            return -1;
        }
    }
}


}  // namespace protocol
}  // namespace blazingdb
