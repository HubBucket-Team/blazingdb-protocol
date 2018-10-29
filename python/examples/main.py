from pygdf import read_csv
import blazingsql

def main():
    column_names = ["n_nationkey", "n_name", "n_regionkey", "n_comments"]
    column_types = ["int32", "int64", "int32", "int64"]
    gdfA = read_csv("data/nation.psv", delimiter='|', dtype=column_types, names=column_names)
    print(gdfA)
    with blazingsql.open_connection('/tmp/orchestrator.socket') as connection:
        print(connection.accessToken)
        db = connection.Database('main')
        tableA = db.Table('nation', gdfA)
        token, unix_path = db.run_query('select id from main.nation', [tableA])
        with db.get_result(token, unix_path) as gdfB:
            print(gdfB)

if __name__ == '__main__':
    main()
