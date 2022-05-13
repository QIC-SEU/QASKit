# QASKit
A Toolkit for Reproducible Study, Application and Verification of Quantum Architecture Search  
# Quantum-circuit-simplification（Simplication.py文件）  
（一）、量子线路进行化简，规则可以自己添加，已经内置五个规则；  
（二）、五个规则分别是：1.刚初始的CNOT门可以删除；2.刚初始的Rz门可以删除；3.两个连续的CNOT门可以删除一个4.两个相同的旋转门可以融合，相位相加；5.超过三个以上的单量子门可以用通用U门表示；  
（三）、还有一个交换规则，Rz可以和CNOT门的控制位交换位置，Rx可以和CNOT门的受控位交换位置；  
（四）、输入的量子现在只支持Rz,Rx和CNOT门，输入线路格式按照Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)]列表形式，参数也按照列表量子门顺序输入，CNOT门没有参数，参数列表比结构列表少所有CNOT门的个数；  
