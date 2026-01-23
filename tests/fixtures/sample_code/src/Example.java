// Example Java file for testing
package com.example;

public class Example {
    private final int value;

    public Example(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static void main(String[] args) {
        Example example = new Example(42);
        System.out.println(example.getValue());
    }
}
