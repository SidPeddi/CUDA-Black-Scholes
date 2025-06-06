#include <iostream>
#include "utils.h"

int main() {
    runBlackScholesCPU();
    runBlackScholesCUDA();
    return 0;
}
