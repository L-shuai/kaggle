import time

def test():
    i = 0
    while True:
        i = i + 1
        if i > 100:
            i = 0
            time.sleep(2)
        print(i)
        if i % 99 == 0:
            print(i)


if __name__ == '__main__':
    test()
