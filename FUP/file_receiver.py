import os
import sys
import socket
import socketserver
import struct

import message
from message import Message

from message_header import Header
from message_body import BodyData
from message_body import BodyRequest
from message_body import BodyResponse
from message_body import BodyResult

from message_util import MessageUtil

CHUNK_SIZE = 4096
upload_dir = ''

class FileReceiveHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print("클라이언트 접속 : {0}".format(self.client_address[0]))

        client = self.request # client socket

        reqMsg = MessageUtil.receive(client) # 클라이언트가 보내온 파일 전송 요청 메세지를 수신한다

        if reqMsg.Header.MSGTYPE != message.REQ_FILE_SEND:
            client.close()
            return

        reqBody = BodyRequest(None)

        print(
            "파일 업로드 요청이 왔습니다. 수락하시겠습니까? yes/no")
        answer = sys.stdin.readline()

        rspMsg = Message()
        rspMsg.Body = BodyResponse(None)
        rspMsg.Body.MSGID = reqMsg.Header.MSGID
        rspMsg.Body.RESPONSE = message.ACCEPTED

        rspMsg.Header = Header(None)

        msgId = 0
        rspMsg.Header.MSGID = msgId
        msgId = msgId + 1
        rspMsg.Header.MSGTYPE = message.REP_FILE_SEND
        rspMsg.Header.BODYLEN = rspMsg.Body.GetSize()
        rspMsg.Header.FRAGMENTED = message.NOT_FRAGMENTED
        rspMsg.Header.LASTMSG = message.LASTMSG
        rspMsg.Header.SEQ = 0

        if answer.strip() != "yes": # 사용자가 'yes'가 아닌 답을 입력하면 클라이언트에게 '거부'응답을 보낸다
            rspMsg.Body = BodyResponse(None)
            rspMsg.Body.MSGID = reqMsg.Header.MSGID
            rspMsg.Body.RESPONSE = message.DENIED
        
            MessageUtil.send(client, rspMsg)
            client.close()
            return
        else:
            MessageUtil.send(client, rspMsg) # 물론'yes'를 입력하면 클라이언트에게 '승낙'응답을 보낸다

            print("파일 전송을 시작합니다...")

            fileSize = reqMsg.Body.FILESIZE
            fileName = reqMsg.Body.FILENAME
            recvFileSize = 0 
            with open(upload_dir + os.sep + "temp.py", 'wb') as file: # 업로드 받을 파일을 생성한다	######################################################## https://freeprog.tistory.com/319 참고 고치기
                dataMsgId = -1
                prevSeq = 0
                
                while True:
                    reqMsg = MessageUtil.receive(client)
                    if reqMsg == None:
                        break

                    print("#", end='')
                    
                    if reqMsg.Header.MSGTYPE != message.FILE_SEND_DATA:
                        break

                    if dataMsgId == -1:
                        dataMsgId = reqMsg.Header.MSGID
                    elif dataMsgId != reqMsg.Header.MSGID:
                        break                    

                    if prevSeq != reqMsg.Header.SEQ: # 메세지 순서가 어긋나면 전송을 중단한다
                        print("{0}, {1}".format(prevSeq, reqMsg.Header.SEQ))
                        break
                    
                    prevSeq += 1

                    recvFileSize += reqMsg.Body.GetSize() # 전송받은 파일의 일부를 담고 있는 bytes 객체를 서버에서 생성한 파일에 기록한다
                    file.write(reqMsg.Body.GetBytes())

                    if reqMsg.Header.LASTMSG == message.LASTMSG: # 마지막 메세지만 반복문을 빠져나온다
                        break
               
                file.close()

                print()
                print("수신 파일 크기 : {0} bytes".format(recvFileSize))

                rstMsg = Message()
                rstMsg.Body = BodyResult(None)
                rstMsg.Body.MSGID = reqMsg.Header.MSGID
                rstMsg.Body.RESULT = message.SUCCESS
                
                rstMsg.Header = Header(None)
                rstMsg.Header.MSGID = msgId
                msgId += 1
                rstMsg.Header.MSGTYPE = message.FILE_SEND_RES
                rstMsg.Header.BODYLEN = rstMsg.Body.GetSize()
                rstMsg.Header.FRAGMENTED = message.NOT_FRAGMENTED
                rstMsg.Header.LASTMSG = message.LASTMSG
                rstMsg.Header.SEQ = 0

                if fileSize == recvFileSize: # 파일 전송 요청에 담겨온 파일 크기와 실제로 받은 파일의 크기를 비교하여 같으면 성공 메세지를 보낸다
                    MessageUtil.send(client, rstMsg)
                else:
                    rstMsg.Body = BodyResult(None)
                    rstMsg.Body.MSGID = reqMsg.Header.MSGID
                    rstMsg.Body.RESULT = message.FAIL
                    MessageUtil.send(client, rstMsg) # 파일 크기에 이상이 있다면 실패 메세지를 보낸다

            print("파일 전송을 마쳤습니다.")                
            client.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법 : {0} <Directory>".format(sys.argv[0]))
        sys.exit(0)

    upload_dir = sys.argv[1]
    if os.path.isdir(upload_dir) == False:
        os.mkdir(upload_dir)
         
    bindPort = 5425
    server = None
    try:
        server = socketserver.TCPServer(
            ('', bindPort), FileReceiveHandler)
            
        print("파일 업로드 서버 시작...")
        server.serve_forever()
    except Exception as err:
        print(err)

    print("서버를 종료합니다.")