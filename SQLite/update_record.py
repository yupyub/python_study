import sqlite3

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute("""
UPDATE PHONEBOOK SET PHONE=?, EMAIL=? WHERE NAME=?
""",("222-222-2222","sinyae222@park.com","PARK SINYE"))

conn.commit()

cursor.execute("""
SELECT NAME, PHONE, EMAIL FROM PHONEBOOK
WHERE NAME=?
""",("PARK SINYE",))  # 튜플이 1개의 요소만 가지는 경우 반드시 뒤에 ","를 붙여야 한다

rows = cursor.fetchall()
for row in rows:
    print("NAME : {0}, PHONE : {1}, EMAIL : {2}".format(row[0],row[1],row[2]))
   
cursor.close()
conn.close()

