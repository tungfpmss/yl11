import sqlite3

conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM users")

# Lấy tất cả kết quả (trả về danh sách các Tuple)
rows = cursor.fetchall()

for row in rows:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}")

conn.close()
