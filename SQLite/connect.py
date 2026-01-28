import sqlite3

# 1. Kết nối (Nếu file chưa tồn tại, nó sẽ tự tạo mới)
conn = sqlite3.connect('my_database.db')

# 2. Tạo đối tượng cursor (con trỏ) để thực thi các câu lệnh SQL
cursor = conn.cursor()

# 3. Tạo bảng
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER
    )
''')

# 4. Lưu thay đổi
conn.commit()

# 5. Đóng kết nối
conn.close()