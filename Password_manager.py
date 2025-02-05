master_password = input("What is the master password? ")


def view():
    pass


def add():
    name = input("Account Name: ")
    password = input("Password: ")

    with open('password.txt', 'a') as f:
        f.write(name + '|' + password)


while True:
    mode = input("Would you like to add a new password or view existing ones (view, add)?\n Or press Q to quit.").lower
    if mode == "q":
        break

    if mode == "view":
        view()
    elif mode == "add":
        add()
    else:
        print("Invalid mode.")
        continue