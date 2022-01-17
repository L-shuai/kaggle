import time
i = 0
while True:
    i=i+1
    if i>100:
        i=0
        time.sleep(1)
    if i%99==0:
        print(i)
    # print(i)