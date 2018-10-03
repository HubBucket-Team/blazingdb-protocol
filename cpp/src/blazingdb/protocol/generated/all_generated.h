// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_ALL_H_
#define FLATBUFFERS_GENERATED_ALL_H_

#include "flatbuffers/flatbuffers.h"

namespace blazingdb {
namespace protocol {
namespace authorization {

struct AuthRequest;

}  // namespace authorization

namespace calcite {

struct DMLRequest;

}  // namespace calcite

namespace flatbuf {
namespace calcite {

struct DDLRequest;

}  // namespace calcite
}  // namespace flatbuf

namespace orchestrator {

struct DMLRequest;

}  // namespace orchestrator

namespace interpreter {

struct DMLRequest;

struct GetResultRequest;

}  // namespace interpreter

struct Header;

struct Request;

namespace authorization {

struct AuthResponse;

}  // namespace authorization

namespace calcite {

struct DMLResponse;

struct DDLResponse;

}  // namespace calcite

namespace orchestrator {

struct DMLResponse;

}  // namespace orchestrator

namespace interpreter {

struct DMLResponse;

struct GetResultResponse;

}  // namespace interpreter

struct Response;

struct ResponseError;

namespace authorization {

enum MessageType {
  MessageType_Auth = 10,
  MessageType_MIN = MessageType_Auth,
  MessageType_MAX = MessageType_Auth
};

inline const MessageType (&EnumValuesMessageType())[1] {
  static const MessageType values[] = {
    MessageType_Auth
  };
  return values;
}

inline const char * const *EnumNamesMessageType() {
  static const char * const names[] = {
    "Auth",
    nullptr
  };
  return names;
}

inline const char *EnumNameMessageType(MessageType e) {
  const size_t index = static_cast<int>(e) - static_cast<int>(MessageType_Auth);
  return EnumNamesMessageType()[index];
}

}  // namespace authorization

namespace calcite {

enum MessageType {
  MessageType_DDL = 0,
  MessageType_DML = 1,
  MessageType_MIN = MessageType_DDL,
  MessageType_MAX = MessageType_DML
};

inline const MessageType (&EnumValuesMessageType())[2] {
  static const MessageType values[] = {
    MessageType_DDL,
    MessageType_DML
  };
  return values;
}

inline const char * const *EnumNamesMessageType() {
  static const char * const names[] = {
    "DDL",
    "DML",
    nullptr
  };
  return names;
}

inline const char *EnumNameMessageType(MessageType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesMessageType()[index];
}

}  // namespace calcite

namespace orchestrator {

enum MessageType {
  MessageType_DDL = 0,
  MessageType_DML = 1,
  MessageType_MIN = MessageType_DDL,
  MessageType_MAX = MessageType_DML
};

inline const MessageType (&EnumValuesMessageType())[2] {
  static const MessageType values[] = {
    MessageType_DDL,
    MessageType_DML
  };
  return values;
}

inline const char * const *EnumNamesMessageType() {
  static const char * const names[] = {
    "DDL",
    "DML",
    nullptr
  };
  return names;
}

inline const char *EnumNameMessageType(MessageType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesMessageType()[index];
}

}  // namespace orchestrator

namespace interpreter {

enum MessageType {
  MessageType_ExecutePlan = 0,
  MessageType_GetResult = 1,
  MessageType_MIN = MessageType_ExecutePlan,
  MessageType_MAX = MessageType_GetResult
};

inline const MessageType (&EnumValuesMessageType())[2] {
  static const MessageType values[] = {
    MessageType_ExecutePlan,
    MessageType_GetResult
  };
  return values;
}

inline const char * const *EnumNamesMessageType() {
  static const char * const names[] = {
    "ExecutePlan",
    "GetResult",
    nullptr
  };
  return names;
}

inline const char *EnumNameMessageType(MessageType e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesMessageType()[index];
}

}  // namespace interpreter

enum Status {
  Status_Error = 0,
  Status_Success = 1,
  Status_MIN = Status_Error,
  Status_MAX = Status_Success
};

inline const Status (&EnumValuesStatus())[2] {
  static const Status values[] = {
    Status_Error,
    Status_Success
  };
  return values;
}

inline const char * const *EnumNamesStatus() {
  static const char * const names[] = {
    "Error",
    "Success",
    nullptr
  };
  return names;
}

inline const char *EnumNameStatus(Status e) {
  const size_t index = static_cast<int>(e);
  return EnumNamesStatus()[index];
}

MANUALLY_ALIGNED_STRUCT(8) Header FLATBUFFERS_FINAL_CLASS {
 private:
  int8_t messageType_;
  int8_t padding0__;  int16_t padding1__;  int32_t padding2__;
  uint64_t payloadLength_;
  uint64_t accessToken_;

 public:
  Header() {
    memset(this, 0, sizeof(Header));
  }
  Header(int8_t _messageType, uint64_t _payloadLength, uint64_t _accessToken)
      : messageType_(flatbuffers::EndianScalar(_messageType)),
        padding0__(0),
        padding1__(0),
        padding2__(0),
        payloadLength_(flatbuffers::EndianScalar(_payloadLength)),
        accessToken_(flatbuffers::EndianScalar(_accessToken)) {
    (void)padding0__;    (void)padding1__;    (void)padding2__;
  }
  int8_t messageType() const {
    return flatbuffers::EndianScalar(messageType_);
  }
  uint64_t payloadLength() const {
    return flatbuffers::EndianScalar(payloadLength_);
  }
  uint64_t accessToken() const {
    return flatbuffers::EndianScalar(accessToken_);
  }
};
STRUCT_END(Header, 24);

namespace authorization {

struct AuthRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           verifier.EndTable();
  }
};

struct AuthRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  explicit AuthRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  AuthRequestBuilder &operator=(const AuthRequestBuilder &);
  flatbuffers::Offset<AuthRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<AuthRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<AuthRequest> CreateAuthRequest(
    flatbuffers::FlatBufferBuilder &_fbb) {
  AuthRequestBuilder builder_(_fbb);
  return builder_.Finish();
}

}  // namespace authorization

namespace calcite {

struct DMLRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_QUERY = 4
  };
  const flatbuffers::String *query() const {
    return GetPointer<const flatbuffers::String *>(VT_QUERY);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_QUERY) &&
           verifier.Verify(query()) &&
           verifier.EndTable();
  }
};

struct DMLRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_query(flatbuffers::Offset<flatbuffers::String> query) {
    fbb_.AddOffset(DMLRequest::VT_QUERY, query);
  }
  explicit DMLRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DMLRequestBuilder &operator=(const DMLRequestBuilder &);
  flatbuffers::Offset<DMLRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DMLRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<DMLRequest> CreateDMLRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> query = 0) {
  DMLRequestBuilder builder_(_fbb);
  builder_.add_query(query);
  return builder_.Finish();
}

inline flatbuffers::Offset<DMLRequest> CreateDMLRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *query = nullptr) {
  return blazingdb::protocol::calcite::CreateDMLRequest(
      _fbb,
      query ? _fbb.CreateString(query) : 0);
}

}  // namespace calcite

namespace flatbuf {
namespace calcite {

struct DDLRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_QUERY = 4
  };
  const flatbuffers::String *query() const {
    return GetPointer<const flatbuffers::String *>(VT_QUERY);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_QUERY) &&
           verifier.Verify(query()) &&
           verifier.EndTable();
  }
};

struct DDLRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_query(flatbuffers::Offset<flatbuffers::String> query) {
    fbb_.AddOffset(DDLRequest::VT_QUERY, query);
  }
  explicit DDLRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DDLRequestBuilder &operator=(const DDLRequestBuilder &);
  flatbuffers::Offset<DDLRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DDLRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<DDLRequest> CreateDDLRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> query = 0) {
  DDLRequestBuilder builder_(_fbb);
  builder_.add_query(query);
  return builder_.Finish();
}

inline flatbuffers::Offset<DDLRequest> CreateDDLRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *query = nullptr) {
  return blazingdb::protocol::flatbuf::calcite::CreateDDLRequest(
      _fbb,
      query ? _fbb.CreateString(query) : 0);
}

}  // namespace calcite
}  // namespace flatbuf

namespace orchestrator {

struct DMLRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_QUERY = 4
  };
  const flatbuffers::String *query() const {
    return GetPointer<const flatbuffers::String *>(VT_QUERY);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_QUERY) &&
           verifier.Verify(query()) &&
           verifier.EndTable();
  }
};

struct DMLRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_query(flatbuffers::Offset<flatbuffers::String> query) {
    fbb_.AddOffset(DMLRequest::VT_QUERY, query);
  }
  explicit DMLRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DMLRequestBuilder &operator=(const DMLRequestBuilder &);
  flatbuffers::Offset<DMLRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DMLRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<DMLRequest> CreateDMLRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> query = 0) {
  DMLRequestBuilder builder_(_fbb);
  builder_.add_query(query);
  return builder_.Finish();
}

inline flatbuffers::Offset<DMLRequest> CreateDMLRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *query = nullptr) {
  return blazingdb::protocol::orchestrator::CreateDMLRequest(
      _fbb,
      query ? _fbb.CreateString(query) : 0);
}

}  // namespace orchestrator

namespace interpreter {

struct DMLRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_LOGICALPLAN = 4
  };
  const flatbuffers::String *logicalPlan() const {
    return GetPointer<const flatbuffers::String *>(VT_LOGICALPLAN);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_LOGICALPLAN) &&
           verifier.Verify(logicalPlan()) &&
           verifier.EndTable();
  }
};

struct DMLRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_logicalPlan(flatbuffers::Offset<flatbuffers::String> logicalPlan) {
    fbb_.AddOffset(DMLRequest::VT_LOGICALPLAN, logicalPlan);
  }
  explicit DMLRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DMLRequestBuilder &operator=(const DMLRequestBuilder &);
  flatbuffers::Offset<DMLRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DMLRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<DMLRequest> CreateDMLRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> logicalPlan = 0) {
  DMLRequestBuilder builder_(_fbb);
  builder_.add_logicalPlan(logicalPlan);
  return builder_.Finish();
}

inline flatbuffers::Offset<DMLRequest> CreateDMLRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *logicalPlan = nullptr) {
  return blazingdb::protocol::interpreter::CreateDMLRequest(
      _fbb,
      logicalPlan ? _fbb.CreateString(logicalPlan) : 0);
}

struct GetResultRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_TOKEN = 4
  };
  const flatbuffers::String *token() const {
    return GetPointer<const flatbuffers::String *>(VT_TOKEN);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_TOKEN) &&
           verifier.Verify(token()) &&
           verifier.EndTable();
  }
};

struct GetResultRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_token(flatbuffers::Offset<flatbuffers::String> token) {
    fbb_.AddOffset(GetResultRequest::VT_TOKEN, token);
  }
  explicit GetResultRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  GetResultRequestBuilder &operator=(const GetResultRequestBuilder &);
  flatbuffers::Offset<GetResultRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<GetResultRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<GetResultRequest> CreateGetResultRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> token = 0) {
  GetResultRequestBuilder builder_(_fbb);
  builder_.add_token(token);
  return builder_.Finish();
}

inline flatbuffers::Offset<GetResultRequest> CreateGetResultRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *token = nullptr) {
  return blazingdb::protocol::interpreter::CreateGetResultRequest(
      _fbb,
      token ? _fbb.CreateString(token) : 0);
}

}  // namespace interpreter

struct Request FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_HEADER = 4,
    VT_PAYLOAD = 6
  };
  const Header *header() const {
    return GetStruct<const Header *>(VT_HEADER);
  }
  const flatbuffers::Vector<uint8_t> *payload() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_PAYLOAD);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<Header>(verifier, VT_HEADER) &&
           VerifyOffset(verifier, VT_PAYLOAD) &&
           verifier.Verify(payload()) &&
           verifier.EndTable();
  }
};

struct RequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_header(const Header *header) {
    fbb_.AddStruct(Request::VT_HEADER, header);
  }
  void add_payload(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> payload) {
    fbb_.AddOffset(Request::VT_PAYLOAD, payload);
  }
  explicit RequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  RequestBuilder &operator=(const RequestBuilder &);
  flatbuffers::Offset<Request> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Request>(end);
    return o;
  }
};

inline flatbuffers::Offset<Request> CreateRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    const Header *header = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> payload = 0) {
  RequestBuilder builder_(_fbb);
  builder_.add_payload(payload);
  builder_.add_header(header);
  return builder_.Finish();
}

inline flatbuffers::Offset<Request> CreateRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const Header *header = 0,
    const std::vector<uint8_t> *payload = nullptr) {
  return blazingdb::protocol::CreateRequest(
      _fbb,
      header,
      payload ? _fbb.CreateVector<uint8_t>(*payload) : 0);
}

namespace authorization {

struct AuthResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_ACCESSTOKEN = 4
  };
  uint64_t accessToken() const {
    return GetField<uint64_t>(VT_ACCESSTOKEN, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_ACCESSTOKEN) &&
           verifier.EndTable();
  }
};

struct AuthResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_accessToken(uint64_t accessToken) {
    fbb_.AddElement<uint64_t>(AuthResponse::VT_ACCESSTOKEN, accessToken, 0);
  }
  explicit AuthResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  AuthResponseBuilder &operator=(const AuthResponseBuilder &);
  flatbuffers::Offset<AuthResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<AuthResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<AuthResponse> CreateAuthResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t accessToken = 0) {
  AuthResponseBuilder builder_(_fbb);
  builder_.add_accessToken(accessToken);
  return builder_.Finish();
}

}  // namespace authorization

namespace calcite {

struct DMLResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_LOGICALPLAN = 4
  };
  const flatbuffers::String *logicalPlan() const {
    return GetPointer<const flatbuffers::String *>(VT_LOGICALPLAN);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_LOGICALPLAN) &&
           verifier.Verify(logicalPlan()) &&
           verifier.EndTable();
  }
};

struct DMLResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_logicalPlan(flatbuffers::Offset<flatbuffers::String> logicalPlan) {
    fbb_.AddOffset(DMLResponse::VT_LOGICALPLAN, logicalPlan);
  }
  explicit DMLResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DMLResponseBuilder &operator=(const DMLResponseBuilder &);
  flatbuffers::Offset<DMLResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DMLResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<DMLResponse> CreateDMLResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> logicalPlan = 0) {
  DMLResponseBuilder builder_(_fbb);
  builder_.add_logicalPlan(logicalPlan);
  return builder_.Finish();
}

inline flatbuffers::Offset<DMLResponse> CreateDMLResponseDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *logicalPlan = nullptr) {
  return blazingdb::protocol::calcite::CreateDMLResponse(
      _fbb,
      logicalPlan ? _fbb.CreateString(logicalPlan) : 0);
}

struct DDLResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           verifier.EndTable();
  }
};

struct DDLResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  explicit DDLResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DDLResponseBuilder &operator=(const DDLResponseBuilder &);
  flatbuffers::Offset<DDLResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DDLResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<DDLResponse> CreateDDLResponse(
    flatbuffers::FlatBufferBuilder &_fbb) {
  DDLResponseBuilder builder_(_fbb);
  return builder_.Finish();
}

}  // namespace calcite

namespace orchestrator {

struct DMLResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_RESULTTOKEN = 4
  };
  const flatbuffers::String *resultToken() const {
    return GetPointer<const flatbuffers::String *>(VT_RESULTTOKEN);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_RESULTTOKEN) &&
           verifier.Verify(resultToken()) &&
           verifier.EndTable();
  }
};

struct DMLResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_resultToken(flatbuffers::Offset<flatbuffers::String> resultToken) {
    fbb_.AddOffset(DMLResponse::VT_RESULTTOKEN, resultToken);
  }
  explicit DMLResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DMLResponseBuilder &operator=(const DMLResponseBuilder &);
  flatbuffers::Offset<DMLResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DMLResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<DMLResponse> CreateDMLResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> resultToken = 0) {
  DMLResponseBuilder builder_(_fbb);
  builder_.add_resultToken(resultToken);
  return builder_.Finish();
}

inline flatbuffers::Offset<DMLResponse> CreateDMLResponseDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *resultToken = nullptr) {
  return blazingdb::protocol::orchestrator::CreateDMLResponse(
      _fbb,
      resultToken ? _fbb.CreateString(resultToken) : 0);
}

}  // namespace orchestrator

namespace interpreter {

struct DMLResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_RESULTTOKEN = 4
  };
  const flatbuffers::String *resultToken() const {
    return GetPointer<const flatbuffers::String *>(VT_RESULTTOKEN);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_RESULTTOKEN) &&
           verifier.Verify(resultToken()) &&
           verifier.EndTable();
  }
};

struct DMLResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_resultToken(flatbuffers::Offset<flatbuffers::String> resultToken) {
    fbb_.AddOffset(DMLResponse::VT_RESULTTOKEN, resultToken);
  }
  explicit DMLResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DMLResponseBuilder &operator=(const DMLResponseBuilder &);
  flatbuffers::Offset<DMLResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DMLResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<DMLResponse> CreateDMLResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> resultToken = 0) {
  DMLResponseBuilder builder_(_fbb);
  builder_.add_resultToken(resultToken);
  return builder_.Finish();
}

inline flatbuffers::Offset<DMLResponse> CreateDMLResponseDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *resultToken = nullptr) {
  return blazingdb::protocol::interpreter::CreateDMLResponse(
      _fbb,
      resultToken ? _fbb.CreateString(resultToken) : 0);
}

struct GetResultResponse FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_NAMES = 4,
    VT_VALUES = 6
  };
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *names() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_NAMES);
  }
  const flatbuffers::Vector<uint8_t> *values() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_VALUES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAMES) &&
           verifier.Verify(names()) &&
           verifier.VerifyVectorOfStrings(names()) &&
           VerifyOffset(verifier, VT_VALUES) &&
           verifier.Verify(values()) &&
           verifier.EndTable();
  }
};

struct GetResultResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_names(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> names) {
    fbb_.AddOffset(GetResultResponse::VT_NAMES, names);
  }
  void add_values(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> values) {
    fbb_.AddOffset(GetResultResponse::VT_VALUES, values);
  }
  explicit GetResultResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  GetResultResponseBuilder &operator=(const GetResultResponseBuilder &);
  flatbuffers::Offset<GetResultResponse> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<GetResultResponse>(end);
    return o;
  }
};

inline flatbuffers::Offset<GetResultResponse> CreateGetResultResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> names = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> values = 0) {
  GetResultResponseBuilder builder_(_fbb);
  builder_.add_values(values);
  builder_.add_names(names);
  return builder_.Finish();
}

inline flatbuffers::Offset<GetResultResponse> CreateGetResultResponseDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *names = nullptr,
    const std::vector<uint8_t> *values = nullptr) {
  return blazingdb::protocol::interpreter::CreateGetResultResponse(
      _fbb,
      names ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*names) : 0,
      values ? _fbb.CreateVector<uint8_t>(*values) : 0);
}

}  // namespace interpreter

struct Response FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_STATUS = 4,
    VT_PAYLOAD = 6
  };
  Status status() const {
    return static_cast<Status>(GetField<int8_t>(VT_STATUS, 0));
  }
  const flatbuffers::Vector<uint8_t> *payload() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_PAYLOAD);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int8_t>(verifier, VT_STATUS) &&
           VerifyOffset(verifier, VT_PAYLOAD) &&
           verifier.Verify(payload()) &&
           verifier.EndTable();
  }
};

struct ResponseBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_status(Status status) {
    fbb_.AddElement<int8_t>(Response::VT_STATUS, static_cast<int8_t>(status), 0);
  }
  void add_payload(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> payload) {
    fbb_.AddOffset(Response::VT_PAYLOAD, payload);
  }
  explicit ResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ResponseBuilder &operator=(const ResponseBuilder &);
  flatbuffers::Offset<Response> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Response>(end);
    return o;
  }
};

inline flatbuffers::Offset<Response> CreateResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    Status status = Status_Error,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> payload = 0) {
  ResponseBuilder builder_(_fbb);
  builder_.add_payload(payload);
  builder_.add_status(status);
  return builder_.Finish();
}

inline flatbuffers::Offset<Response> CreateResponseDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    Status status = Status_Error,
    const std::vector<uint8_t> *payload = nullptr) {
  return blazingdb::protocol::CreateResponse(
      _fbb,
      status,
      payload ? _fbb.CreateVector<uint8_t>(*payload) : 0);
}

struct ResponseError FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_ERRORS = 4
  };
  const flatbuffers::String *errors() const {
    return GetPointer<const flatbuffers::String *>(VT_ERRORS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ERRORS) &&
           verifier.Verify(errors()) &&
           verifier.EndTable();
  }
};

struct ResponseErrorBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_errors(flatbuffers::Offset<flatbuffers::String> errors) {
    fbb_.AddOffset(ResponseError::VT_ERRORS, errors);
  }
  explicit ResponseErrorBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ResponseErrorBuilder &operator=(const ResponseErrorBuilder &);
  flatbuffers::Offset<ResponseError> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ResponseError>(end);
    return o;
  }
};

inline flatbuffers::Offset<ResponseError> CreateResponseError(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> errors = 0) {
  ResponseErrorBuilder builder_(_fbb);
  builder_.add_errors(errors);
  return builder_.Finish();
}

inline flatbuffers::Offset<ResponseError> CreateResponseErrorDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *errors = nullptr) {
  return blazingdb::protocol::CreateResponseError(
      _fbb,
      errors ? _fbb.CreateString(errors) : 0);
}

namespace authorization {

}  // namespace authorization

namespace calcite {

}  // namespace calcite

namespace flatbuf {
namespace calcite {

}  // namespace calcite
}  // namespace flatbuf

namespace orchestrator {

}  // namespace orchestrator

namespace interpreter {

}  // namespace interpreter

namespace authorization {

}  // namespace authorization

namespace calcite {

}  // namespace calcite

namespace orchestrator {

}  // namespace orchestrator

namespace interpreter {

}  // namespace interpreter

}  // namespace protocol
}  // namespace blazingdb

#endif  // FLATBUFFERS_GENERATED_ALL_H_
