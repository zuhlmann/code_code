#From docs.python tutorial I believe  ZRU 2/27/2019

def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        #nonlocal will make spam a nonlocal variable ZRU
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)
