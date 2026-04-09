import sys
sys.path.insert(0, '.')

def test_asi():
    from l104_asi.quantum import QuantumComputationCore
    qc = QuantumComputationCore()
    # Test repetition code
    result = qc.quantum_error_correct(code_type='repetition', error_type='bit_flip', shots=100)
    assert result['quantum'] == True
    assert 'logical_error_rate' in result
    assert 0 <= result['logical_error_rate'] <= 1
    print('ASI QEC repetition test passed:', result)
    # Test surface code placeholder
    result2 = qc.quantum_error_correct(code_type='surface', error_type='depolarizing', shots=50)
    assert result2['quantum'] == True
    print('ASI QEC surface test passed:', result2)
    return True

def test_agi():
    from l104_agi.core import AGICore
    agi = AGICore()
    result = agi.quantum_error_correction_score(code_type='repetition', shots=200)
    assert 'logical_error_rate' in result
    print('AGI QEC wrapper test passed:', result)
    return True

if __name__ == '__main__':
    try:
        test_asi()
        test_agi()
        print('All QEC integration tests passed.')
    except Exception as e:
        print('Test failed:', e)
        sys.exit(1)