import sqlite3

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute("""
DELETE FROM PHONEBOOK WHERE EMAIL=?
""",("bumsu@kim.com",))
conn.commit()

cursor.execute("SELECT * FROM PHONEBOOK")

rows = cursor.fetchall()
for row in rows:
    print(row)
    
cursor.close()
conn.close()