import sqlite3

conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

# Dữ liệu mẫu
user_data = ("Gemini", 1)

# Thêm một dòng
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", user_data)

# Thêm nhiều dòng cùng lúc
many_users = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
cursor.executemany("INSERT INTO users (name, age) VALUES (?, ?)", many_users)

conn.commit()
conn.close()
