import pennylane as qml
from pennylane import numpy as np
import operator


#判断量子线路中量子们相邻量子们的列表标记和连接的具体细节
def neighbor(ansatz,order):
    '''
    :param ansatz:输入的量子线路结构，列表格式; Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :param order: 所查找量子门所处量子线路结构的列表标记; Example:第一个量子门，所处列表标记为0
    :return:Control_Bit:初始为[-1,-1],单比特门只考虑这个，列表的第一位是前面相邻门的序号，两比特门是控制位对应前面序号，列表第二位是后面相邻门序号，两比特类似,
            Controlled_Bit:初始为[-1，-1]，两比特门受控位对应相邻门的序号，列表第一位是前面相邻列表的序号，列表第二位是后面相邻门序号,
            Quantum_Gate_Marking:初始为[0，0，0，0]，共四位的列表，前两个代表控制比特前后邻居门的类型，1表示单比特门，2表示两比特门的控制位，3表示两比特门的受控位，后两个表示受控比特的前后邻居门的类型;
    '''
    Control_Bit = [-1,-1]
    Controlled_Bit = [-1,-1]
    Quantum_Gate_Marking = [0,0,0,0]

    #判断Rx,Rz前后的邻居编号
    if ansatz[order][0] == 'rx' or ansatz[order][0] == 'rz':
        #寻找前面邻居编号
        if order == 0:
            Control_Bit[0] = -1
            Controlled_Bit[0] = -1
        else:
            for i in range(order-1,-1,-1):
                if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 1
                        break
                elif ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 3
                        break
                else:
                    print('出现未知量子门')
                    break
        #寻找后面邻居编号
        if order == len(ansatz):
            Control_Bit[1] = -1
            Controlled_Bit[1] = -1
        else:
            for i in range(order+1,len(ansatz),1):
                if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 1
                        break
                elif ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 3
                        break
                else:
                    print('出现未知量子门')
                    break

    #判断Cx前后的邻居编号
    elif ansatz[order][0] == 'cx':
        #寻找前面控制比特的邻居编号
        if order == 0:
            Control_Bit[0] = -1
            Controlled_Bit[0] = -1
        else:
            for i in range(order-1,-1,-1):
                if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 3
                        break
        #寻找后面控制比特的邻居编号
        if order == len(ansatz):
            Control_Bit[1] = -1
            Controlled_Bit[1] = -1
        else:
            for i in range(order+1,len(ansatz),1):
                if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 3
                        break
        #寻找前面受控制比特的邻居编号
        if order == 0:
            Control_Bit[0] = -1
            Controlled_Bit[0] = -1
        else:
            for i in range(order-1,-1,-1):
                if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[0] = i
                        Quantum_Gate_Marking[2] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[0] = i
                        Quantum_Gate_Marking[2] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][2]:
                        Controlled_Bit[0] = i
                        Quantum_Gate_Marking[2] = 3
                        break
        #寻找后面受控制比特的邻居编号
        if order == len(ansatz):
            Control_Bit[1] = -1
            Controlled_Bit[1] = -1
        else:
            for i in range(order+1,len(ansatz),1):
                if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[1] = i
                        Quantum_Gate_Marking[3] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[1] = i
                        Quantum_Gate_Marking[3] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][2]:
                        Controlled_Bit[1] = i
                        Quantum_Gate_Marking[3] = 3
                        break

    else:
        print('查找的是未知的量子门')
    return Control_Bit,Controlled_Bit,Quantum_Gate_Marking

#搜寻量子线路中固定列表标记再参数列表中的下标,CNOT门的下标是其前面旋转门参数的下标
def Search_parameters(ansatz, order):
    new_order = 0
    for i in range(0,order,1):
        if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
            new_order = new_order + 1
    return new_order


#删除初始的CNOT门
def rule_1(ansatz):
    '''
    :param ansatz: 输入的量子线路结构，列表格式; Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :return: 使用规则一后的量子线路结构;
    '''
    new_ansatz = list(ansatz)
    Count = len(new_ansatz)
    i = 0
    while Count > 0:
        if new_ansatz[i][0] == 'cx':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[0] == -1 and Controlled_Bit[0] == -1:
                new_ansatz.pop(i)
                i = i - 1
        i = i + 1
        Count = Count - 1
    print('使用规则1次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz

#删除初始的Rz门
def rule_2(ansatz, parameters):
    new_ansatz = list(ansatz)
    new_parameters = list(parameters)
    Count = len(new_ansatz)
    i = 0
    while Count > 0:
        if new_ansatz[i][0] == 'rz':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[0] == -1:
                new_parameters.pop(Search_parameters(new_ansatz,i))
                new_ansatz.pop(i)
                i = i - 1
        i = i + 1
        Count = Count - 1
    print('使用规则2次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, new_parameters

#删除两个重复的CNOT门中前面一个
def rule_3(ansatz):
    '''
    :param ansatz: 输入的量子线路结构，列表格式; Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :return: 使用规则一后的量子线路结构;
    '''
    new_ansatz = list(ansatz)
    Count = len(new_ansatz)
    i = 0
    while Count > 0:
        if new_ansatz[i][0] == 'cx':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[1] == Controlled_Bit[1] and Control_Bit[1] != -1:
                new_ansatz.pop(i)
                i = i - 1
        i = i + 1
        Count = Count - 1
    print('使用规则3次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz

#删除两个相同的旋转门中前面一个，相位相加
def rule_4(ansatz, parameters):
    new_ansatz = list(ansatz)
    new_parameters = list(parameters)
    Count = len(new_ansatz)
    i = 0
    while Count > 0:
        Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
        if new_ansatz[i][0] == 'rx':
            if Control_Bit[1] != -1 and new_ansatz[Control_Bit[1]][0] == 'rx':
                new_parameters[Search_parameters(new_ansatz,Control_Bit[1])] += new_parameters[Search_parameters(new_ansatz,i)]
                new_parameters.pop(Search_parameters(new_ansatz,i))
                new_ansatz.pop(i)
                i = i - 1
        elif new_ansatz[i][0] == 'rz':
            if Control_Bit[1] != -1 and new_ansatz[Control_Bit[1]][0] == 'rz':
                new_parameters[Search_parameters(new_ansatz,Control_Bit[1])] += new_parameters[Search_parameters(new_ansatz,i)]
                new_parameters.pop(Search_parameters(new_ansatz,i))
                new_ansatz.pop(i)
                i = i - 1
        i = i + 1
        Count = Count - 1
    print('使用规则4次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, new_parameters

#删除超出三个的连续单比特门，其中优先删除后面几个，相位直接删除，其他相位保持不变
def rule_5(ansatz, parameters):
    new_ansatz = list(ansatz)
    new_parameters = list(parameters)
    Count = len(new_ansatz)
    Bit_String = [0 for x in range(len(new_ansatz)+1)]
    Condition = ['rx', 'rz']
    Condition1 = ['rz', 'rx']
    i = 0
    while Count > 0:
        if new_ansatz[i][0] == 'rx':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if new_ansatz[Control_Bit[1]][0] == 'rz':
                Control_Bit1, Controlled_Bit1, Quantum_Gate_Marking1 = neighbor(new_ansatz, Control_Bit[1])
                if new_ansatz[Control_Bit1[1]][0] == 'rx':
                    Control_Bit2, Controlled_Bit2, Quantum_Gate_Marking2 = neighbor(new_ansatz, Control_Bit1[1])
                    if new_ansatz[Control_Bit2[1]][0] == 'rz':
                        Bit_String[0] = i
                        Bit_String[1] = Control_Bit[1]
                        Bit_String[2] = Control_Bit1[1]
                        Bit_String[3] = Control_Bit2[1]
                        j = 3
                        while new_ansatz[Bit_String[j]][0] == Condition[j%2]:
                            Control_Bit3, Controlled_Bit3, Quantum_Gate_Marking3 = neighbor(new_ansatz, Bit_String[j])
                            j = j + 1
                            Bit_String[j] = Control_Bit3[1]
                        Bit_String[j] = -1

                        for k in range(0,j-3,1):
                            new_parameters.pop(Search_parameters(new_ansatz,Bit_String[k+3])-k)
                            new_ansatz.pop(Bit_String[k+3]-k)
                            Count = Count - 1
        elif new_ansatz[i][0] == 'rz':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if new_ansatz[Control_Bit[1]][0] == 'rx':
                Control_Bit1, Controlled_Bit1, Quantum_Gate_Marking1 = neighbor(new_ansatz, Control_Bit[1])
                if new_ansatz[Control_Bit1[1]][0] == 'rz':
                    Control_Bit2, Controlled_Bit2, Quantum_Gate_Marking2 = neighbor(new_ansatz, Control_Bit1[1])
                    if new_ansatz[Control_Bit2[1]][0] == 'rx':
                        Bit_String[0] = i
                        Bit_String[1] = Control_Bit[1]
                        Bit_String[2] = Control_Bit1[1]
                        Bit_String[3] = Control_Bit2[1]
                        j = 3
                        while new_ansatz[Bit_String[j]][0] == Condition1[j % 2]:
                            Control_Bit3, Controlled_Bit3, Quantum_Gate_Marking3 = neighbor(new_ansatz, Bit_String[j])
                            j = j + 1
                            Bit_String[j] = Control_Bit3[1]
                        Bit_String[j] = -1

                        for k in range(0, j - 3, 1):
                            new_parameters.pop(Search_parameters(new_ansatz, Bit_String[k + 3]) - k)
                            new_ansatz.pop(Bit_String[k + 3] - k)
                            Count = Count - 1
        Count = Count - 1
        i = i + 1
    print('使用规则5次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, new_parameters

#单比特门Rz、Rx向左移动
def rule_Zshift_left(ansatz, parameters):
    new_ansatz = list(ansatz)
    new_parameters = list(parameters)
    for i in range(0,len(new_ansatz),1):
        if new_ansatz[i][0] == 'rx':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[0] != -1 and  Quantum_Gate_Marking[0] == 3:
                new_parameters.insert(Search_parameters(new_ansatz,Control_Bit[0]),new_parameters[Search_parameters(new_ansatz,i)])
                new_parameters.pop(Search_parameters(new_ansatz, i+1))
                new_ansatz.insert(Control_Bit[0],new_ansatz[i])
                new_ansatz.pop(i+1)

        elif new_ansatz[i][0] == 'rz':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[0] != -1 and  Quantum_Gate_Marking[0] == 2:
                new_parameters.insert(Search_parameters(new_ansatz,Control_Bit[0]),new_parameters[Search_parameters(new_ansatz,i)])
                new_parameters.pop(Search_parameters(new_ansatz, i+1))
                new_ansatz.insert(Control_Bit[0],new_ansatz[i])
                new_ansatz.pop(i+1)

    return new_ansatz,new_parameters

#按照规则，化简量子线路
def simplification(ansatz, parameters):
    #测试输入
    count = 0
    for i in range(len(ansatz)):
        if ansatz[i][0] == 'rx' or ansatz[i][0] == 'rz':
            count +=1
    if count != len(parameters):
        print('输入结构和相位不匹配')
    else:
        new_ansatz = list(ansatz)
        new_parameters = list(parameters)
        symbol = 0
        while symbol == 0:
            ansatz_copy = list(new_ansatz)
            parameters_copy = list(new_parameters)
            new_ansatz, new_parameters = rule_Zshift_left(new_ansatz,new_parameters)
            new_ansatz = rule_1(new_ansatz)
            new_ansatz, new_parameters = rule_2(new_ansatz,new_parameters)
            new_ansatz = rule_3(new_ansatz)
            new_ansatz, new_parameters = rule_4(new_ansatz, new_parameters)
            new_ansatz, new_parameters = rule_5(new_ansatz, new_parameters)
            if operator.eq(ansatz_copy,new_ansatz) and operator.eq(parameters_copy,new_parameters):
                symbol = 1

        circle = len(new_ansatz)
        k = 0
        while k < len(new_ansatz):
            if new_ansatz[k][0] == 'rx':
                Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, k)
                if Control_Bit[1] != -1 and Quantum_Gate_Marking[1] == 3:
                    new_parameters.insert(Search_parameters(new_ansatz, Control_Bit[1]),new_parameters[Search_parameters(new_ansatz, k)])
                    new_parameters.pop(Search_parameters(new_ansatz, k))
                    new_ansatz.insert(Control_Bit[1], new_ansatz[k])
                    new_ansatz.pop(k)
                    symbol1 = 0
                    while symbol1 == 0:
                        ansatz_copy = list(new_ansatz)
                        parameters_copy = list(new_parameters)
                        new_ansatz = rule_1(new_ansatz)
                        new_ansatz, new_parameters = rule_2(new_ansatz, new_parameters)
                        new_ansatz = rule_3(new_ansatz)
                        new_ansatz, new_parameters = rule_4(new_ansatz, new_parameters)
                        new_ansatz, new_parameters = rule_5(new_ansatz, new_parameters)
                        if operator.eq(ansatz_copy, new_ansatz) and operator.eq(parameters_copy, new_parameters):
                            symbol1 = 1
                        else:
                            k = 0

            elif new_ansatz[k][0] == 'rz':
                Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, k)
                if Control_Bit[1] != -1 and Quantum_Gate_Marking[1] == 2:
                    new_parameters.insert(Search_parameters(new_ansatz, Control_Bit[1]),new_parameters[Search_parameters(new_ansatz, k)])
                    new_parameters.pop(Search_parameters(new_ansatz, k))
                    new_ansatz.insert(Control_Bit[1], new_ansatz[k])
                    new_ansatz.pop(k)
                    symbol1 = 0
                    while symbol1 == 0:
                        ansatz_copy = list(new_ansatz)
                        parameters_copy = list(new_parameters)
                        new_ansatz = rule_1(new_ansatz)
                        new_ansatz, new_parameters = rule_2(new_ansatz, new_parameters)
                        new_ansatz = rule_3(new_ansatz)
                        new_ansatz, new_parameters = rule_4(new_ansatz, new_parameters)
                        new_ansatz, new_parameters = rule_5(new_ansatz, new_parameters)
                        if operator.eq(ansatz_copy, new_ansatz) and operator.eq(parameters_copy, new_parameters):
                            symbol1 = 1
                        else:
                            k = 0
            k = k + 1

    #测试输出
    count = 0
    for i in range(len(new_ansatz)):
        if new_ansatz[i][0] == 'rx' or new_ansatz[i][0] == 'rz':
            count = count + 1
    if count != len(new_parameters):
        print('输入结构和相位不匹配')
    return new_ansatz, new_parameters


'''Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)];'''
lst= [('rz', 0), ('rx', 3),('cx', 0, 1), ('cx', 2, 3),('rx', 0), ('cx', 1, 2),('rz', 1), ('rx', 3), ('rx',2),('rx', 2)]
lst1= [1,2,3,4,5,6,7]
a,b = simplification(lst,lst1)
print(a,b)