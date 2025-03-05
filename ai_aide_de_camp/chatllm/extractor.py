

class Filter:
    def __init__(self, message):
        self.message = message


    def process(self, task, pattern=None):
        if task == "extract":
            if "type" in self.message:
                if pattern == "type":
                    return self.message["type"]

                if self.message["type"] == "msg::single":  # 微信私聊
                    if pattern == "from_wxid":
                        return self.message['data']["from_wxid"]
                    if pattern == 'from_nickname':
                        return self.message['data']['from_nickname']
                    if pattern == 'msg':
                        return self.message['data']['msg']
                    if pattern == 'send_or_recv':
                        return self.message['data']['send_or_recv']
                    if pattern == 'time':
                        return self.message['data']['time']
                    if pattern == "data_type":
                        return self.message["data"]["data_type"]

                if self.message['type'] == 'msg::chatroom':  # 微信群聊
                    if pattern == 'from_chatroom_wxid':
                        return self.message['data']['from_chatroom_wxid']
                    if pattern == 'msg':
                        return self.message['data']['msg']
                    if pattern == 'send_or_recv':
                        return self.message['data']['send_or_recv']
                    if pattern == 'from_member_wxid':
                        return self.message['data']['from_member_wxid']
                    if pattern == 'time':
                        return self.message['data']['time']

                if self.message['type'] == 'friend::person':  # 微信好友
                    if pattern == 'wx_id':
                        return self.message['data']['wx_id']
                    if pattern == 'wx_id_search':
                        return self.message['data']['wx_id_search']
                    if pattern == 'wx_nickname':
                        return self.message['data']['wx_nickname']
                    if pattern == 'wx_avatar':
                        return self.message['data']['wx_avatar']
                    if pattern == 'remark_name':
                        return self.message['data']['remark_name']

                if self.message['type'] == 'friend::gh':  # 微信公众号
                    if pattern == 'gh_id':
                        return self.message['data']['gh_id']
                    if pattern == 'gh_id_search':
                        return self.message['data']['gh_id_search']
                    if pattern == 'gh_name':
                        return self.message['data']['gh_name']
                    if pattern == 'gh_avatar':
                        return self.message['data']['gh_avatar']

                if self.message['type'] == 'friend::chatroom':  # 群信息
                    if pattern == 'chatroom_id':
                        return self.message['data']['chatroom_id']
                    if pattern == 'chatroom_name':
                        return self.message['data']['chatroom_name']
                    if pattern == 'chatroom_avatar':
                        return self.message['data']['chatroom_avatar']

                if self.message['type'] == 'member::chatroom':  # 群成员信息
                    if pattern == 'chatroom_id':
                        return self.message['data']['chatroom_id']
                    if pattern == 'wx_id':
                        return self.message['data']['wx_id']
                    if pattern == 'wx_id_search':
                        return self.message['data']['wx_id_search']
                    if pattern == 'wx_nickname':
                        return self.message['data']['wx_nickname']

        if task == "classify":
            if "type" in self.message:
                return self.message["type"]


        return None

if __name__ == "__main__":
    msg = {'user': 'wxid_9tdxaat685zt21', 'type': 'msg::single', 'data': {'data_type': '1', 'send_or_recv': '1+[Phone]', 'from_wxid': 'filehelper', 'time': '2025-03-05 10:11:31', 'msg': '1', 'msg_byte_hex': '31', 'from_nickname': 'File Transfer'}}
    filter = Filter(msg)
    result1 = filter.process("extract","from_wxid")
    print("result1:", result1)
    result2 = filter.process("classify")
    print("result2:", result2)

