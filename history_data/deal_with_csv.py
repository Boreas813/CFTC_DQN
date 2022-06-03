import os

def add_title():
    for i in os.listdir(os.getcwd()):
        try:
            if i == 'deal_with_csv.py':
                continue
            if 'CFTC' in i:
                continue
            with open(i, 'r+') as file:
                content = file.read()
                if not content.startswith('date_time'):
                    print(f'正在为{i}添加表头')
                    file.seek(0, 0)
                    file.write('date_time,minute,open,high,low,close,volume\n'+content)
        except Exception as e:
            print(f'{i} 出现异常')


if __name__ == '__main__':
    add_title()
