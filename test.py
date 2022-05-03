from math import log
from unittest import result
from sklearn.ensemble import RandomForestClassifier
def countVowels(string): #统计函数
    vowels = "aeiouAEIOU"
    count = 0
    for i in string: #循环遍历
        if i in vowels: 
            count += 1  #计数
    return count/len(string) #返回统计结果
def countNumber(string): #统计函数
    vowels = "1234567890"
    count = 0
    for i in string: #循环遍历
        if i in vowels: 
            count += 1  #计数
    return count/len(string) #返回统计结果
def InfoEntropy(str_1):
    Info_map = {}

    for i in str_1:
        #   不统计的字符↓
        if i != ' ' and i != '"' and i != "." and i != ',':
            if i in Info_map.keys():
                Info_map[i] += 1
            else:
                Info_map[i] = 1

    return calcShannonEnt(Info_map)


def calcShannonEnt(dataSet):
    numEntries = 0
    shannonEnt = 0.0

    for key in dataSet:
        numEntries += dataSet[key]

    # 计算信息熵
    for key in dataSet:
        prob = float(dataSet[key]) / numEntries  # 计算p(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


featureMatrix = []
labelList = []
print("Initialize Matrix")
with open("train.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or line =="":
            continue
        tokens = line.split(",")
        name = tokens[0]
        target = name.split(".")
        label = tokens[1]

        entropy=InfoEntropy(target[0])
        num_count=countNumber(target[0])
        vowels_count=countVowels(target[0])
        lenth_domain=len(target[0])
        segement_count=len(target)

        featureMatrix.append([entropy,num_count,vowels_count,lenth_domain,segement_count])
        if label == "notdga":
            labelList.append(0)
        else:
            labelList.append(1)

print("Begin Training")
clf = RandomForestClassifier(random_state=0)
clf.fit(featureMatrix,labelList)

#test
test_domain_name=[]
testMatrix=[]
with open("train.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or line =="":
            continue
        tokens = line.split(",")
        name = tokens[0]
        test_domain_name.append(name)
        target = name.split(".")

        entropy=InfoEntropy(target[0])
        num_count=countNumber(target[0])
        vowels_count=countVowels(target[0])
        lenth_domain=len(target[0])
        segement_count=len(target)

        testMatrix.append([entropy,num_count,vowels_count,lenth_domain,segement_count])
f.close()


print("Begin Predicting")
test_result_matrix=clf.predict(testMatrix)
f=open('result.txt','w')
for i in range(0,len(testMatrix)):
    f.write(test_domain_name[i]+',')
    if(test_result_matrix[i]==0):
        f.write('notdga\n')
    else:
        f.write('dga\n')
f.close()
