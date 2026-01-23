// Example C++ file for testing
#include <iostream>
#include <memory>

class Example {
public:
    explicit Example(int value) : value_(value) {}

    int getValue() const { return value_; }

private:
    int value_;
};

int main() {
    auto example = std::make_unique<Example>(42);
    std::cout << example->getValue() << std::endl;
    return 0;
}
