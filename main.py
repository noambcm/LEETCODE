import array
import numpy as np

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n=len(grid)
        for i,liste in enumerate(grid):
            if liste.count(1)==n-1:
                return i



class Solution:
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        dictionnaire={'P':-1, 'G':-1, 'M':-1}
        paper=0
        metal=0
        glass=0
        for i,s in enumerate(garbage):
            if 'P' in s:
                paper+=s.count('P')
                dictionnaire['P']=i
            if 'G' in s:
                glass+=s.count('G')
                dictionnaire['G']=i
            if 'M' in s:
                metal+=s.count('M')
                dictionnaire['M']=i
        somme=paper+glass+metal
        for key in dictionnaire:
            if dictionnaire[key]!=-1:
                for j in range(dictionnaire[key]):
                    somme+=travel[j]
        return somme
    


class Solution:
    def minOperations(self, n: int) -> int:
        arr=[0]*n
        for i in range(n):
            arr[i]=2*i+1
        if n%2!=0:
            somme=0
            for i in range(n//2):
                somme+=arr[n//2]-arr[i]
            return somme
        else:
            somme=1
            arr[n//2]=arr[n//2 - 1]+1
            arr[n//2 - 1]=arr[n//2]
            for i in range(n//2):
                somme+=arr[n//2]-arr[i]
            return somme



class Solution:
    def first_positive_index(self, nums:List[int])->int:
        left=0
        right=len(nums)-1
        while left<=right:
            middle=(left+right)//2
            if nums[middle]<=0:
                left=middle+1
            else:
                right=middle-1
        return left

    def last_negative_index(self, nums:List[int]) ->int:
        left=0
        right=len(nums)-1
        while left<=right:
            middle=(left+right)//2
            if nums[middle]<0:
                left=middle+1
            else:
                right=middle-1
        return right

    def maximumCount(self, nums: List[int]) -> int:
        positive_index=self.first_positive_index(nums)
        negative_index=self.last_negative_index(nums)
        pos=len(nums)-positive_index
        neg=negative_index+1
        return max(pos,neg)
    


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left=0
        right=len(nums)-1
        while left<=right:
            middle=(left+right)//2
            if nums[middle]==target:
                return middle
            elif nums[middle]<target:
                left=middle+1
            else:
                right=middle-1
        return -1
    

#Erreur de code pcq j'update a chaque fois copy_list. Peu efficace mais l'idee est la
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        copy_list=list(arr)
        n=len(arr)
        i=0
        while i<n:
            if arr[i]==0 and i+2<n:
                arr[i+1]=0
                for j in range(i+2,n):
                    arr[j]=copy_list[j-1]
                i+=2
                copy_list=list(arr)
            if arr[i]==0 and i+2>=n:
                arr[i+1]=0
            else:
                i+=1

#Version corrigee
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        n=len(arr)
        result=[]
        for num in arr:
            if num!=0:
                result.append(num)
            if num==0:
                result.append(num)
                if len(result)<n:
                    result.append(0)
            if len(result)>=n:
                break
        for i in range(n):
            arr[i]=result[i]



class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        occurence=defaultdict(int)
        a=0
        b=0
        for liste in grid:
            for element in liste:
                occurence[element]+=1
        for i in range(1,(len(grid[0]))**2 + 1):
            if i not in occurence:
                b=i
                break
        a=max(occurence, key=occurence.get)
        return [a,b]
    


class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        indices=defaultdict(list)
        if k==0:
            if 0 in nums:
                return False
            return True
        for i,num in enumerate(nums):
            indices[num].append(i)
        liste=indices[1]
        n=len(liste)
        for i in range(n):
            for j in range(n):
                if i!=j and abs(liste[i]-liste[j])-1<k:
                    return False
        return True
    





class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        output=[]
        x=map(list,itertools.permutations(nums))
        return list(x)
    

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n=len(nums)
        indices=defaultdict(list)
        for i,num in enumerate(nums):
            indices[num].append(i)
        liste_indices=[]
        for j in range(1,k+1):
            liste_indices.append(max(indices[j]))
        return n-min(liste_indices)
    


class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        if len(set(target))!=len(target):
            dict_s=defaultdict(int)
            dict_target=defaultdict(int)
            for char in s:
                dict_s[char]+=1
            for char in target:
                dict_target[char]+=1
            liste=[]
            for char in target:
                liste.append(dict_s[char]//dict_target[char])
            return min(liste)
        liste_occurence=[]
        for char in target:
            liste_occurence.append(s.count(char))
        return min(liste_occurence)



class Solution:
    def largestDigit(self,num:int) -> int:
        s=str(num)
        maximum=0
        for char in s:
            maximum=max(maximum, int(char))
        return maximum

    def maximumSum(self,nums:List[int])->int:
        n=len(nums)
        if nums==[]:
            return 0
        maximum=0
        for i in range(n):
            for j in range(n):
                if j!=i:
                    maximum=max(maximum,nums[i]+nums[j])
        return maximum


    def maxSum(self, nums: List[int]) -> int:
        max_digit=defaultdict(list)
        for num in nums:
            largest_digit_num=self.largestDigit(num)
            max_digit[largest_digit_num].append(num)
        liste_somme=[]
        for liste in max_digit.values():
            if len(liste)>1:
                liste_somme.append(self.maximumSum(liste))
        if liste_somme==[]:
            return -1
        return max(liste_somme)
    

class Solution:
    def minSteps(self, s: str, t: str) -> int:
        count_s=Counter(s)
        count_t=Counter(t)
        steps=0
        for char in count_s:
            if char in count_t:
                steps+=max(0,count_s[char]-count_t[char])
            else:
                steps+=count_s[char]
        return steps
    


class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        mot1=sorted(list(set(word1)))
        mot2=sorted(list(set(word2)))
        if mot1!=mot2:
            return False  
        occurence1=defaultdict(int)
        occurence2=defaultdict(int)
        for char in word1:
            occurence1[char]+=1
        for char in word2:
            occurence2[char]+=1
        liste1=sorted(list(occurence1.values()))
        liste2=sorted(list(occurence2.values()))
        return liste1==liste2
    


#Erreur Buddy Strings je me suis casse le cerveau
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s)!=len(goal):
            return False
        if len(goal)==len(s)==1:
            return False
        mot1=sorted(list(set(s)))
        mot2=sorted(list(set(goal)))
        if mot1!=mot2:
            return False
        indices_s=defaultdict(list)
        indices_goal=defaultdict(list)
        for i,char in enumerate(s):
            indices_s[char].append(i)
        for i,char in enumerate(goal):
            indices_goal[char].append(i)
        if s!=goal:
            count=0
            for char in s:
                if indices_s[char]!=indices_goal[char]:
                    count+=1
            return count==2
        if s==goal:
            if len(list(set(s)))==1:
                return True
            numbers = [x for x in range(2,10)]
            lengths = [len(liste) for liste in indices_s.values()]
            intersection = set(numbers) & set(lengths)
            if intersection:
                return True
            else:
                return False
            
#Correction
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s)!=len(goal):
            return False
        if s==goal:
            seen=set()
            for char in s:
                if char in seen:
                    return True
                seen.add(char)
            return False
        if s!=goal:
            n=len(s)
            liste=[]
            for i in range(n):
                if s[i]!=goal[i]:
                    liste.append(i)
            if len(liste)==2 and s[liste[0]]==goal[liste[1]] and s[liste[1]]==goal[liste[0]]:
                return True
            return False
        



class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        if s1==s2:
            return True
        diff=[]
        for i in range(len(s1)):
            if s1[i]!=s2[i]:
                diff.append(i)
        if len(diff)==2 and s1[diff[0]]==s2[diff[1]] and s1[diff[1]]==s2[diff[0]]:
            return True
        return False
    







 

