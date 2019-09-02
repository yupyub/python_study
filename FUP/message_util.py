import socket

import message
from message import Message
from message_header import Header
from message_body import BodyRequest
from message_body import BodyResponse
from message_body import BodyData
from message_body import BodyResult


class MessageUtil:
    
    # send() 메소드는 msg 매개변수가 담고 있는 모든 바이트를 내보낼 때까지 반복해서 socket.send() 메소드를 호출한다
    @staticmethod
    def send(sock, msg): 
        sent = 0
        buffer = msg.GetBytes()
        while sent < msg.GetSize():            
            sent += sock.send(buffer)

    @staticmethod
    def receive(sock):
        totalRecv = 0
        sizeToRead = 16 # 헤더의 크기
        hBuffer = bytes() # 헤더 버퍼

        # 헤더 읽기
        while sizeToRead > 0: # 첫 반복문에서는 스트림으로부터 메세지 헤더의 경계를 끊어낸다      
            buffer = sock.recv(sizeToRead)
            if len(buffer) == 0:
                return None

            hBuffer += buffer
            totalRecv += len(buffer)
            sizeToRead -= len(buffer)

        header = Header(hBuffer)

        totalRecv = 0
        bBuffer = bytes()
        sizeToRead = header.BODYLEN

        while sizeToRead > 0: # 첫 반복문에서 얻은 헤더에서 본문의 길이를 뽑아내어, 그 길이만큼 다시 스트림으로부터 본문을 읽는다
            buffer = sock.recv(sizeToRead)
            if len(buffer) == 0:
                return None

            bBuffer += buffer
            totalRecv += len(buffer)
            sizeToRead -= len(buffer)

        body = None

        if header.MSGTYPE == message.REQ_FILE_SEND:
            body = BodyRequest(bBuffer)
        elif header.MSGTYPE == message.REP_FILE_SEND:
            body = BodyResponse(bBuffer)
        elif header.MSGTYPE == message.FILE_SEND_DATA:
            body = BodyData(bBuffer)
        elif header.MSGTYPE == message.FILE_SEND_RES:
            body = BodyResult(bBuffer)
        else:
            raise Exception(                     
                    "Unknown MSGTYPE : {0}".
                    format(header.MSGTYPE))

        msg = Message()
        msg.Header = header
        msg.Body = body
            
        return msg
    