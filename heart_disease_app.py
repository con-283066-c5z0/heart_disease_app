#Name of table
table_name = "heart_table"

#Reopened the connection to the SQLite database
connection = sqlite3.connect("heart.db")

#Get data from db to df
query = f"SELECT * FROM {table_name};"
df = pd.read_sql(query, connection)

#Closed the connection
connection.close()
