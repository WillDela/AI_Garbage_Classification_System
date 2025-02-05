print("Welcome to my computer quiz!")

playing = input("Do you want to play? ")

if playing.lower() != "yes":
    quit()

print("Okay! Lets play ")

score = 0

answer = input("What does CPU stand for? ").lower()
if answer == "central":
    print("Correct!")
    score += 1
else:
    print("Incorrect!")

answer = input("What does RAM stand for? ").lower()
if answer == "random":
    print("Correct!")
    score += 1
else:
    print("Incorrect!")

answer = input("What does gpu stand for? ").lower()
if answer == "graphics":
    print("Correct!")
    score +=1
else:
    print("Incorrect!")

answer = input("What does W stand for? ").lower()
if answer == "william":
    print("Correct!")
    score+=1
else:
    print("Incorrect!")

print("You got " + str(score) +  " out of 4 correct!")
print("You got " + str((score / 4) * 100) +  "%.")