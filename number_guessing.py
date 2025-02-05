import random

highest_val = input("Enter a range: ")

if highest_val.isdigit():
    highest_val = int(highest_val)

    if highest_val <= 0:
        print("Type a number larger than 0.")
        quit()
else:
    print("Please type a number next time.")
    quit()

random_number = random.randint(0, highest_val)
guesses: int = 0

while True:
    guesses += 1
    user_guess = input("Guess a number in 10 or less tries: ")
    if user_guess.isdigit():
        user_guess = int(user_guess)
    else:
        print("Please type a number.")
        continue
    if user_guess == random_number:
        print("You got it!")
        print("You got it in " + str(guesses) + " guesses!")
        break
    elif user_guess < random_number:
        print("Guess a higher number.")
    else:
        print("Guess a lower number.")

    if guesses > 10:
        print("Too many guesses, try again!")
        break