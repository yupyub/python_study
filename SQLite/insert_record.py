import sqlite3
conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute("""
INSERT INTO PHONEBOOK (NAME,PHONE, EMAIL)
VALUES(?,?,?)
""", ("PARK SINYE","021-322-1234","sinyae@park.com"))

id = cursor.lastrowid
print(id)

cursor.execute("""
INSERT INTO PHONEBOOK (NAME,PHONE, EMAIL)
VALUES(?,?,?)
""", ("김범수","021-322-1234","bumsu@kim.com"))

id = cursor.lastrowid
print(id)

conn.commit()
cursor.close()
conn.close()
