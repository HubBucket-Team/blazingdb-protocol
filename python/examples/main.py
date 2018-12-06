import pandas as pd
import cudf as cudf
import blazingsql

def main():
    column_names = ['n_nationkey', 'n_name', 'n_regionkey', 'n_comments']
    column_types = {'n_nationkey': 'int32', 'n_regionkey': 'int64'}
    nation_df = pd.read_csv("data/nation.psv", delimiter='|', dtype=column_types, names=column_names)
    nation_df = nation_df[['n_nationkey', 'n_regionkey']]

    print(nation_df)

    with blazingsql.open_connection('/tmp/orchestrator.socket') as connection:
        print(connection.accessToken)
        db = connection.Database('main')

        nation_gdf = cudf.DataFrame.from_pandas(nation_df)
        tableA = db.Table('nation', nation_gdf)
        token, unix_path = db.run_query('select id from main.nation', [tableA])
        with db.get_result(token, unix_path) as gdfB:
            print(gdfB)

if __name__ == '__main__':
    main()
