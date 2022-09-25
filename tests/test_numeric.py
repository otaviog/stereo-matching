from stereomatch.numeric import next_power_of_2

def test_should_return_next_power_2():
    assert next_power_of_2(0) == 1
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(4) == 4
    assert next_power_of_2(50) == 64
    assert next_power_of_2(127) == 128
