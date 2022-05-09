with open(r'C:\Users\X\PycharmProjects\XProject\mac.txt', 'w') as f:
    str_centre = "/gps/pos/centre "
    str_run = "/run/beamOn "
    runTime = 10000000
    # for x in range(0, 0, 1000) :
    # for z in range(0, 0, 0) :
    for y in range(150, 650, 1) :
        print(y)
        f.write(str_centre + str(0) + ' ' + str(y) + ' ' + str(0) + ' mm' + '\r')
        f.write(str_run + str(runTime) + '\r')
print("finish11ed")