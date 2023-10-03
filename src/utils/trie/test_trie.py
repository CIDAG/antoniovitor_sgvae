import unittest
from . import Trie

class TestTrie(unittest.TestCase):
    def setUp(self) -> None:
        self.trie = Trie()
    
    def test_basis_operations(self):
        # assert insertion without data
        self.trie.set('ABCDEF')
        self.assertTrue(self.trie.has('ABCDEF'))
        
        # assert insertion with data
        self.trie.set('test', 'data')
        self.assertEqual(self.trie.get('test'), 'data')

        
    # def test_creation_with_initial_elements(self):
    #     elements = [
    #         'abcdef',
    #         '123456',
    #         'xyzxyz',
    #     ]

    #     trie = Trie(list=elements)
    #     results = [trie.has()]
    #     self.assertTrue(results)

if __name__ == '__main__':
    unittest.main()