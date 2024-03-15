import unittest

from parts_analysis import PartsAnalysis

analyzer = PartsAnalysis()


class TestParseString(unittest.TestCase):
    def setUp(self):
        self.parser = analyzer

    def test_watt_extraction(self):
        text = ["This is a 50W device"]
        expected = {'productName': [], 'watt': '50W'}
        self.assertEqual(self.parser.parse_string(text), expected)

    def test_ohm_extraction(self):
        text = ["Resistance is 10kohm"]
        expected = {'productName': [], 'ohm': '10kohm'}
        self.assertEqual(self.parser.parse_string(text), expected)

    def test_voltage_extraction(self):
        text = ["250V is the voltage"]
        expected = {'productName': [], 'voltage': '250V'}
        self.assertEqual(self.parser.parse_string(text), expected)

        text = ["TLV70225DBV"]
        expected = {'productName': ['TLV70225DBV']}
        self.assertEqual(self.parser.parse_string(text), expected)

    def test_no_match(self):
        text = ["Just a random sentence"]
        expected = {'productName': [text[0]]}
        self.assertEqual(self.parser.parse_string(text), expected)

    def test_size_extraction(self):
        test_cases = [
            (["10"], {'productName': ['10']}),
            (["20.3"], {'productName': ['20.3']}),
            # 기본
            (["10X20"], {'productName': [], 'size': '10X20'}),
            (["10X20X30"], {'productName': [], 'size': '10X20X30'}),
            (["1.2X2.33"], {'productName': [], 'size': '1.2X2.33'}),
            (["1.2X2.33X5.25"], {'productName': [], 'size': '1.2X2.33X5.25'}),
            # 앞에 붙은 문자열
            (["CAP10X20"], {'productName': ['CAP10X20']}),
            (["ABC10X20X30"], {'productName': ['ABC10X20X30']}),
            (["ERF1.2X2.33"], {'productName': ['ERF1.2X2.33']}),
            (["RTA1.2X2.33X5.25"], {'productName': ['RTA1.2X2.33X5.25']}),
            # 앞에 붙은 문자열, 뒤에 공백 문자열
            (["CAP10X20 RTA"], {'productName': ['CAP10X20 RTA']}),
            (["ABC10X20X30 DFG"], {'productName': ['ABC10X20X30 DFG']}),
            (["ERF1.2X2.33 WER"], {'productName': ['ERF1.2X2.33 WER']}),
            (["RTA1.2X2.33X5.25 RGA"], {'productName': ['RTA1.2X2.33X5.25 RGA']}),
            # 앞에 공백 문자열
            (["CAP 10X20"], {'productName': [], 'size': '10X20'}),
            (["CAP 10X20X30"], {'productName': [], 'size': '10X20X30'}),
            (["CAP 1.2X2.33"], {'productName': [], 'size': '1.2X2.33'}),
            (["CAP 1.2X2.33X5.25"], {'productName': [], 'size': '1.2X2.33X5.25'}),
            # 앞에 공백 문자열, 뒤에 공백 문자열
            (["CAP 10X20 ERF"], {'productName': [], 'size': '10X20'}),
            (["CAP 10X20X30 EEF"], {'productName': [], 'size': '10X20X30'}),
            (["CAP 1.2X2.33 ASD"], {'productName': [], 'size': '1.2X2.33'}),
            (["CAP 1.2X2.33X5.25 VVD"], {'productName': [], 'size': '1.2X2.33X5.25'}),
            # 대소문자
            (["10x20"], {'productName': [], 'size': '10x20'}),
            (["10x20x30"], {'productName': [], 'size': '10x20x30'}),
            (["1.2x2.33"], {'productName': [], 'size': '1.2x2.33'}),
            (["1.2x2.33x5.25"], {'productName': [], 'size': '1.2x2.33x5.25'}),
            # mm 포함
            (["10X20mm"], {'productName': [], 'size': '10X20mm'}),
            (["10X20X30mm"], {'productName': [], 'size': '10X20X30mm'}),
            (["1.2X2.33mm"], {'productName': [], 'size': '1.2X2.33mm'}),
            (["1.2X2.33X5.25mm"], {'productName': [], 'size': '1.2X2.33X5.25mm'}),
            (["10X20mm10"], {'productName': ['10X20mm10']}),
            (["10X20mmab"], {'productName': ['10X20mmab']}),
            (["10X20mm SPOS"], {'productName': [], 'size': '10X20mm'}),
            # mm, 앞에 붙은 문자열
            (["CAP10X20mm"], {'productName': ['CAP10X20mm']}),
            (["ABC10X20X30mm"], {'productName': ['ABC10X20X30mm']}),
            (["ERF1.2X2.33mm"], {'productName': ['ERF1.2X2.33mm']}),
            (["RTA1.2X2.33X5.25mm"], {'productName': ['RTA1.2X2.33X5.25mm']}),
            # mm, 앞에 붙은 문자열, 뒤에 공백 문자열
            (["CAP10X20mm RTA"], {'productName': ['CAP10X20mm RTA']}),
            (["ABC10X20X30mm DFG"], {'productName': ['ABC10X20X30mm DFG']}),
            (["ERF1.2X2.33mm WER"], {'productName': ['ERF1.2X2.33mm WER']}),
            (["RTA1.2X2.33X5.25mm RGA"], {'productName': ['RTA1.2X2.33X5.25mm RGA']}),
            # mm, 앞에 공백 문자열
            (["CAP 10X20mm"], {'productName': [], 'size': '10X20mm'}),
            (["CAP 10X20X30mm"], {'productName': [], 'size': '10X20X30mm'}),
            (["CAP 1.2X2.33mm"], {'productName': [], 'size': '1.2X2.33mm'}),
            (["CAP 1.2X2.33X5.25mm"], {'productName': [], 'size': '1.2X2.33X5.25mm'}),
            # mm, 앞에 공백 문자열, 뒤에 공백 문자열
            (["CAP 10X20mm ERF"], {'productName': [], 'size': '10X20mm'}),
            (["CAP 10X20X30mm EEF"], {'productName': [], 'size': '10X20X30mm'}),
            (["CAP 1.2X2.33mm ASD"], {'productName': [], 'size': '1.2X2.33mm'}),
            (["CAP 1.2X2.33X5.25mm VVD"], {'productName': [], 'size': '1.2X2.33X5.25mm'}),
            # 사이즈 포함
            (["10X20사이즈"], {'productName': [], 'size': '10X20사이즈'}),
            (["10X20X30사이즈"], {'productName': [], 'size': '10X20X30사이즈'}),
            (["1.2X2.33사이즈"], {'productName': [], 'size': '1.2X2.33사이즈'}),
            (["1.2X2.33X5.25사이즈"], {'productName': [], 'size': '1.2X2.33X5.25사이즈'}),
            # size, 앞에 붙은 문자열
            (["CAP10X20사이즈"], {'productName': ['CAP10X20사이즈']}),
            (["ABC10X20X30사이즈"], {'productName': ['ABC10X20X30사이즈']}),
            (["ERF1.2X2.33사이즈"], {'productName': ['ERF1.2X2.33사이즈']}),
            (["RTA1.2X2.33X5.25사이즈"], {'productName': ['RTA1.2X2.33X5.25사이즈']}),
            # size, 앞에 붙은 문자열, 뒤에 공백 문자열
            (["CAP10X20사이즈 RTA"], {'productName': ['CAP10X20사이즈 RTA']}),
            (["ABC10X20X30사이즈 DFG"], {'productName': ['ABC10X20X30사이즈 DFG']}),
            (["ERF1.2X2.33사이즈 WER"], {'productName': ['ERF1.2X2.33사이즈 WER']}),
            (["RTA1.2X2.33X5.25사이즈 RGA"], {'productName': ['RTA1.2X2.33X5.25사이즈 RGA']}),
            # size, 앞에 공백 문자열
            (["CAP 10X20사이즈"], {'productName': [], 'size': '10X20사이즈'}),
            (["CAP 10X20X30사이즈"], {'productName': [], 'size': '10X20X30사이즈'}),
            (["CAP 1.2X2.33사이즈"], {'productName': [], 'size': '1.2X2.33사이즈'}),
            (["CAP 1.2X2.33X5.25사이즈"], {'productName': [], 'size': '1.2X2.33X5.25사이즈'}),
            # size, 앞에 공백 문자열, 뒤에 공백 문자열
            (["CAP 10X20사이즈 ERF"], {'productName': [], 'size': '10X20사이즈'}),
            (["CAP 10X20X30사이즈 EEF"], {'productName': [], 'size': '10X20X30사이즈'}),
            (["CAP 1.2X2.33사이즈 ASD"], {'productName': [], 'size': '1.2X2.33사이즈'}),
            (["CAP 1.2X2.33X5.25사이즈 VVD"], {'productName': [], 'size': '1.2X2.33X5.25사이즈'}),
            # mm만 있음
            (["10mm"], {'productName': [], 'size': '10mm'}),
            (["22.0mm"], {'productName': [], 'size': '22.0mm'}),
            (["RCCS10mm"], {'productName': ['RCCS10mm']}),
            (["RCCS22.0mm"], {'productName': ['RCCS22.0mm']}),
            (["10MM"], {'productName': [], 'size': '10MM'}),
            (["22.0MM"], {'productName': [], 'size': '22.0MM'}),
            (["RCCS10MM"], {'productName': ['RCCS10MM']}),
            (["RCCS22.0MM"], {'productName': ['RCCS22.0MM']}),
            (["10mmca"], {'productName': ['10mmca']}),
            (["22.0mmca"], {'productName': ['22.0mmca']}),
            (["RCCS10mmda"], {'productName': ['RCCS10mmda']}),
            (["RCCS22.0mmda"], {'productName': ['RCCS22.0mmda']}),
            (["10MMCA"], {'productName': ['10MMCA']}),
            (["22.0MMCA"], {'productName': ['22.0MMCA']}),
            (["RCCS10MMDA"], {'productName': ['RCCS10MMDA']}),
            (["RCCS22.0MMDA"], {'productName': ['RCCS22.0MMDA']}),
            (["10mm SDE"], {'productName': [], 'size': '10mm'}),
            (["22.0mm WEF"], {'productName': [], 'size': '22.0mm'}),
            (["10MM EDC"], {'productName': [], 'size': '10MM'}),
            (["22.0MM CDC"], {'productName': [], 'size': '22.0MM'}),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                self.assertEqual(self.parser.parse_string(text), expected)

    # 필요한 경우 여기에 추가 테스트 케이스를 작성할 수 있습니다.
    def test_etc_extraction(self):
        # text = ['Capacitor', 'MLCC', '102R18W103KV4E', '3216', 'CAP', '1206', '3216']
        # expected = {'productName': [text[0]]}
        # self.assertEqual(self.parser.parse_string(text, 'C'), expected)

        text = ['Resistor', '52.3K', 'RES', '0402', '1005', '추가']
        expected = {'productName': ['Resistor', 'RES', '0402', '추가'], 'ohm': '52.3KOhm', 'size': '1005'}
        self.assertEqual(self.parser.parse_string(text, 'R'), expected)

    def test_is_part_number(self):
        # Test the function with the provided examples
        product_names = [
            # "PS2801C-4-F3-A", "102R18W103KV4E", "QS5244TQ", "RC0603FR-07330KL",
            # "SN74LVC2G32QDCURQ1", "TPS62050DGSR", "TPS7B6750QPWPRQ1", "TSW-105-07-L-D-LL",
            # "UPD78F0546GC(R)-UBT-A", "1N5230B-TAP", "SS18", "MIS-2"
            # "2.4MHz", "Step-Down", "Switching", "Regulator",
            # "DCDC", "Step", "Down",
            "MP2143DJ", "TSOT23-8", "TSOT23-8", "MP2161", "MPS", "U13U21U22", "100", "946-MP2143DJLFZ"
        ]

        # Applying the function to each product name
        is_part_number = [analyzer.is_part_number(name) for name in product_names]
        print(is_part_number)

if __name__ == '__main__':
    unittest.main()
