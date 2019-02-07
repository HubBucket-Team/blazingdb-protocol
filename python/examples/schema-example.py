from blazingdb.protocol.orchestrator import NodeConnectionSchema
from blazingdb.protocol.orchestrator import DMLResponseSchema, DMLDistributedResponseSchema


def main():
    z = NodeConnectionSchema(path='sam', port=10, type=20)
    a = DMLResponseSchema(resultToken=11, nodeConnection=z, calciteTime=20)
    b = DMLResponseSchema.From(a.ToBuffer())
    print(b.resultToken)

    x = DMLDistributedResponseSchema(size=2, responses=[a])
    y = DMLDistributedResponseSchema.From(x.ToBuffer())

    col = list(item for item in y.responses)
    for xx in col:
        print(xx.calciteTime)


if __name__ == '__main__':
    main()
