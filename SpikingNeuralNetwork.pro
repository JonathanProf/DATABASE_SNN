TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        functions.cpp \
        main.cpp

HEADERS += \
    functions.h

DISTFILES += \
    BD/AeAi.csv \
    BD/AiAe.csv \
    BD/BD400/XeAe.csv \
    BD/XeAe.csv \
    BD/assignments.csv \
    BD/inputS_784_35.csv \
    BD/inputSpikesPoisson.csv \
    BD/proportions.csv \
    BD/theta.csv \
    BD/theta_A.csv
