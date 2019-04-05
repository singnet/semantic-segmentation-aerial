import sys
import os
import time

test = None


def change():
    global test
    print(test)
    test = 12
    print(test)
    return


def main():
    start = time.time()
    # print(sys.executable)
    # print(sys.argv)
    while True:
        if test == 12:
            os.execv(sys.executable, ['python3'] + sys.argv)
        time.sleep(0.1)
        if (time.time() - start) >= 7:
            change()


if __name__ == "__main__":
    main()