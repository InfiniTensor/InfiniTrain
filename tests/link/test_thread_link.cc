#include <thread>

int main() {
  int value = 0;
  std::thread worker([&value] { value = 1; });
  worker.join();
  return value == 1 ? 0 : 1;
}
