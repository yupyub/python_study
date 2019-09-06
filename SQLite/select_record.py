import sqlite3

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute("SELECT NAME, PHONE, EMAIL FROM PHONEBOOK") # "SELECT * FROM PHONEBOOK" 

#row1 = cursor.fetchone()
#print(row1)

rows = cursor.fetchall()
for row in rows:
    print("NAME : {0}, PHONE : {1}, EMAIL : {2}".format(row[0],row[1],row[2]))
    
cursor.close()
conn.close()
    
    