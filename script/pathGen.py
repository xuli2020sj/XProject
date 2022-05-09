import numpy as np

def string_generate():
    with open(r'C:\Users\X\PycharmProjects\XProject\9mac.txt', 'w') as f:
        str_centre = "/gps/pos/centre "
        str_run = "/run/beamOn "
        str_energy = "/gps/energy "
        runTime = 100000000
        # for x in range(0, 0, 1000) :
        # for energy in [46.52, 59.54,88.03,122.1,165.85,391.69,661.7,898.06,1173.24,1332.5,1836.08] :
        for y in range(500, 1500, 5):
            # f.write(str_energy + str(energy) + ' keV' + '\r')
            f.write(str_centre + str(1219) + ' ' + str(y) + ' ' + str(665.5) + ' mm' + '\r')
            f.write(str_run + str(runTime) + '\r')

            

if __name__ == '__main__':
    gridNumOfX = 5
    gridNumOfY = 2
    gridNumOfZ = 3

    XLength = 2438
    YLength = 1926
    ZLength = 1331

    xCellLength = XLength / gridNumOfX
    xHalfCellLength = XLength / gridNumOfX/2
    yCellLength = YLength / gridNumOfY
    yHalfCellLength = YLength / gridNumOfY/2
    zCellLength = ZLength / gridNumOfZ
    zHalfCellLength = ZLength / gridNumOfZ/2

    xPos = np.arange(0, XLength, XLength / gridNumOfX).reshape(1, gridNumOfX)
    xPos += xCellLength / 2

    yPos = np.arange(0, YLength, YLength / gridNumOfY).reshape(1, gridNumOfY)
    yPos += yCellLength / 2 + 500

    zPos = np.arange(0, ZLength, ZLength / gridNumOfZ).reshape(1, gridNumOfZ)
    zPos += zCellLength / 2

    pos = np.zeros((gridNumOfX, gridNumOfY, gridNumOfZ, 3))

    # 网格中心坐标
    for x in range(0, gridNumOfX):
        for y in range(0, gridNumOfY):
            for z in range(0, gridNumOfZ):
                pos[x][y][z][0] = xPos[0][x]
                pos[x][y][z][1] = yPos[0][y]
                pos[x][y][z][2] = zPos[0][z]

    # 运动路径生成
    # for x in range(0, gridNumOfX):
    #     for z in range(0, gridNumOfZ):
    #         print('x', pos[x][z][0], 'z', pos[x][z][1])