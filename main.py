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
    


class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        restaurant=defaultdict(list)
        for i,string in enumerate(list1):
            if string in list2:
                restaurant[string].append(i)
                restaurant[string].append(list2.index(string))
        output=[]
        minimum=min([sum(liste) for liste in restaurant.values()])
        for key,val in restaurant.items():
            if sum(val)==minimum:
                output.append(key)
        return output


class Solution:
    def secondHighest(self, s: str) -> int:
        numbers=[int(char) for char in s if char.isdigit()]
        if len(list(set(numbers)))<2:
            return -1
        unique_numbers=list(set(numbers))
        liste_triee=sorted(unique_numbers, reverse=True)
        return liste_triee[1]
    
    
class Solution:
    def removeDigit(self, number: str, digit: str) -> str:
        occurence=defaultdict(list)
        for i,char in enumerate(number):
            occurence[char].append(i)
        elu=0
        liste_indice=occurence[digit]
        for i in liste_indice:
            candidat=int(number[:i]+number[i+1:])
            elu=max(elu,candidat)
        return str(elu)
    

class Solution:
    def areOccurrencesEqual(self, s: str) -> bool:
        occurence=defaultdict(int)
        for char in s:
            occurence[char]+=1
        return len(list(set(occurence.values())))==1
    


class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        frequency=defaultdict(int)
        for num in nums:
            frequency[num]+=1
        m=max(list(frequency.values()))
        liste_num=[]
        for key,val in frequency.items():
            if val==m:
                liste_num.append(key)
        somme=0
        for num in liste_num:
            somme+=nums.count(num)
        return somme
    


class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        n=len(nums1)
        m=len(nums2)
        pair=0
        for i in range(n):
            for j in range(m):
                if nums1[i]%(nums2[j]*k)==0:
                    pair+=1
        return pair



class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        n=len(nums)
        pair=0
        for i in range(n):
            for j in range(i+1,n):
                if abs(nums[i]-nums[j])==k:
                    pair+=1
        return pair


class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        n=len(nums)
        pair=0
        for i in range(n):
            for j in range(i+1,n):
                if nums[i]==nums[j] and (i*j)%k==0:
                    pair+=1
        return pair
    


class Solution:
    def sumDigit(self,n:int)->int:
        somme=0
        for char in str(n):
            somme+=int(char)
        return somme 

    def countLargestGroup(self, n: int) -> int:
        occurence=defaultdict(int)
        for number in range(1,n+1):
            sum_number=self.sumDigit(number)
            occurence[sum_number]+=1
        max_values=max(occurence.values())
        return len([key for key,val in occurence.items() if val==max_values])
    


class Solution:
    def numDifferentIntegers(self, word: str) -> int:
        seen=set()
        n=len(word)
        i=0
        while i<n:
            if word[i].isdigit():
                number=word[i]
                while i+1<n and word[i+1].isdigit():
                    number+=word[i+1]
                    i+=1
                seen.add(int(number))
            i+=1
        return len(seen)
    


class Solution:
    def commonLetters(self, word1:str,word2:str)->bool:
        for char in word1:
            if char in word2:
                return False
        return True 

    def maxProduct(self, words: List[str]) -> int:
        maximum=0
        for i,word in enumerate(words):
            for j in range(i+1,len(words)):
                if self.commonLetters(word,words[j]):
                    maximum=max(maximum,len(word)*len(words[j]))
        return maximum
    


class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        n=len(arr)
        left=0
        right=n-1
        while left<=right:
            middle=(left+right)//2
            if arr[middle+1]-arr[middle]>=0:
                left=middle+1
            else:
                right=middle-1
        return left
    

class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num+2*t
    

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        if not [num for num in nums if num%3!=0]:
            return 0
        somme=0
        for i in range(len(nums)):
            if nums[i]%3==2:
                somme+=1
            else:
                somme+=nums[i]%3
        return somme
    

class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        string=str(num)
        potentielle=[]
        beauty=0
        for i in range(len(string)-k+1):
            potentielle.append(int(string[i:i+k]))
        for candidat in potentielle:
            if candidat!=0 and num%candidat==0:
                beauty+=1
        return beauty
    

class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        if n==1:
            return 0
        position=0
        direction=1
        for _ in range(k):
            position+=direction
            if position==0 or position==n-1:
                direction*=-1
        return position
    


class Solution:
    def passThePillow(self, n: int, time: int) -> int:
        position=1
        direction=1
        for _ in range(time):
            position+=direction
            if position==1 or position==n:
                direction*=-1
        return position
    


class Solution:
    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        if purchaseAmount%10==0:
            return 100-purchaseAmount
        basis=(purchaseAmount//10)*10
        if purchaseAmount%10>=5:
            basis+=10
        return 100-basis
    


class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count=0
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if gcd(int(str(nums[i])[0]),int(str(nums[j])[-1]))==1:
                    count+=1
        return count
    



class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count=0
        for integer in range(low,high+1):
            x=str(integer)
            if len(x)%2==0:
                n=len(x)//2
                first_half=[int(x[i]) for i in range(0,n)]
                second_half=[int(x[i]) for i in range(n,len(x))]
                if sum(first_half)==sum(second_half):
                    count+=1
        return count
    


class Solution:
    def sumDigit(self, n:int)->int:
        string=str(n)
        n=len(string)
        somme=0
        for i in range(n):
            somme+=int(string[i])
        return somme 

    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        dictionnaire={i:0 for i in range(1,46)}
        for number in range(lowLimit, highLimit+1):
            sum_number=self.sumDigit(number)
            dictionnaire[sum_number]+=1
        return max(list(dictionnaire.values()))
    


class Solution:
    def countTriples(self, n: int) -> int:
        count=0
        for a in range(1,n):
            for b in range(1,n):
                x=a*a+b*b
                racine=int(sqrt(x))
                if racine*racine==x and racine<=n:
                    count+=1
        return count
    


class Solution:
    def kItemsWithMaximumSum(self, numOnes: int, numZeros: int, numNegOnes: int, k: int) -> int:
        x=min(numOnes,k)      
        restant1=max(0,k-x) 
        y=min(numZeros,restant1)
        restant2=max(0,restant1-y)
        max_sum=x+restant2*(-1)
        return max_sum 
    


class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        liste=sorted(nums)
        n=len(nums)
        counting={'positif':0, 'negatif':0}
        for num in nums:
            if num>=0:
                counting['positif']+=1
            counting['negatif']+=1
        if counting['negatif']==0 or counting['positif']==0:
            return liste[n-1]*liste[n-2]*liste[n-3]
        candidat1=liste[n-1]*liste[n-2]*liste[n-3]
        candidat2=liste[0]*liste[1]*liste[-1]
        return max(candidat1,candidat2)
    


class Solution:
    def gcd(self,a:int,b:int)->int:
        while b>0:
            a,b=b,a%b
        return a
    
    def divisor(self,n:int)->int:
        divisors=[]
        for i in range(1,n+1):
            if n%i==0:
                divisors.append(i)
        return divisors
        
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        len1=len(str1)
        len2=len(str2)
        gcd_length=self.gcd(len1,len2)
        divisors=sorted(self.divisor(gcd_length), reverse=True)
        for d in divisors:
            candidat=str1[:d]
            if candidat*(len1//d)==str1 and candidat*(len2//d)==str2:
                return candidat
        return ""
    


#Premiere version nzive ---> Time Limit
class Solution:
    def isMonotone(self,n:int)->bool:
        number=str(n)
        length=len(number)
        i=0
        while i<length-1:
            if int(number[i+1])<int(number[i]):
                return False
            i+=1
        return True

    def monotoneIncreasingDigits(self, n: int) -> int:
        for i in range(n,-1,-1):
            if self.isMonotone(i):
                return i
            

#Deuxieme version
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        number=list(str(n))
        length=len(number)
        mark=length
        for i in range(length-1):
            if int(number[i])>int(number[i+1]):
                mark=i
                break
        if mark==length:
            return n
        while mark>0 and number[mark]==number[mark-1]:
            mark-=1
        number[mark]=str(int(number[mark])-1)
        for i in range(mark+1,length):
            number[i]='9'
        return int("".join(number))



class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        number=min(x,y//4)
        if number%2!=0:
            return "Alice"
        return "Bob"
    

class Solution:
    def numberOfCuts(self, n: int) -> int:
        if n==1:
            return 0
        if n%2==0:
            return n//2
        return n
    

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        string1=list(reversed(a))
        string2=list(reversed(b))
        somme1=0
        somme2=0
        for i in range(len(string1)):
            somme1+=int(string1[i])*(2**i)
        for i in range(len(string2)):
            somme2+=int(string2[i])*(2**i)
        somme=somme1+somme2
        return bin(somme)[2:]
    



class Solution:
    def toGoatLatin(self, sentence: str) -> str:
        liste_sortie=[]
        list_words=sentence.split()
        for i,word in enumerate(list_words):
            if word[0].lower() in ['a','e','i','o','u']:
                liste_sortie.append(word+'ma'+'a'*(i+1))
            else:
                word=word[1:] + word[0]
                liste_sortie.append(word+'ma'+'a'*(i+1))
        return ' '.join(liste_sortie)
    


class Solution:
    def diviseurs(self,num:int)-> list:
        liste_diviseurs=[]
        for i in range(1,num//2+1):
            if num%i==0:
                liste_diviseurs.append(i)
        return liste_diviseurs

    def checkPerfectNumber(self, num: int) -> bool:
        if num%2!=0:
            return False
        return sum(self.diviseurs(num))==num
    

#Dans la base n-2, n=1x(n-2)^1 + 2x(n-2)^0 donc sa representation dans la base n-2 est 12. 1é n'est pas un palindrome
class Solution:
    def isStrictlyPalindromic(self, n: int) -> bool:
        return False
    

class Solution:
    def reverseInteger(self,n:int)->int:
        string=str(n)
        reversed_string=string[::-1]
        return int(reversed_string)

    def countDistinctIntegers(self, nums: List[int]) -> int:
        liste_reversed=[0]*len(nums)
        for i,number in enumerate(nums):
            liste_reversed[i]=self.reverseInteger(number)
        liste_sortie=nums + liste_reversed
        return len(set(liste_sortie))
    


class Solution:
    def reverse(self, x: int) -> int:
        string=str(abs(x))
        reversed_string=string[::-1]
        reversed_int=int(reversed_string)
        if not -(2**31)<=reversed_int<=2**31-1:
            return 0
        if x>=0:
            return reversed_int
        else:
            return -reversed_int
        



class Solution:
    def reversal(self,num:int)->int:
        string=str(num)
        reversed_string=string[::-1]
        reversed_integer=int(reversed_string)
        return reversed_integer

    def isSameAfterReversals(self, num: int) -> bool:
        first_reversed=self.reversal(num)
        second_reversed=self.reversal(first_reversed)
        return num==second_reversed
    


class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        count=0
        for i in range(len(words)-1):
            for j in range(i+1,len(words)):
                if words[i]==words[j][::-1]:
                    count+=1
        return count
    


class Solution:
    def manipulation(self,nums:List[int])->list:
        liste_sortie=[0]*(len(nums)-1)
        for i in range(len(nums)-1):
            liste_sortie[i]=(nums[i]+nums[i+1])%10
        return liste_sortie

    def triangularSum(self, nums: List[int]) -> int:
        while len(nums)>1:
            nums=self.manipulation(nums)
        return nums[0]
    


class Solution:
    def round(self,s:str,k:int)->str:
        n=len(s)
        first_list=[]
        i=0
        while i<len(s):
            first_list.append(s[i:i+k])
            i+=k
        second_list=[0]*len(first_list)
        for i,element in enumerate(first_list):
            second_list[i]=sum([int(char) for char in element])
        sortie=str()
        for i in range(len(second_list)):
            sortie=sortie + str(second_list[i])
        return sortie

    def digitSum(self, s: str, k: int) -> str:
        if len(s)>k:
            sortie=self.round(s,k)
        while len(s)>k:
            s=self.round(s,k)
        return s
    



class Solution:
    def operation(self,nums:List[int])->List[int]:
        n=len(nums)
        newNums=[0]*(n//2)
        for i in range(n//2):
            if i%2==0:
                newNums[i]=min(nums[2*i],nums[2*i+1])
            else:
                newNums[i]=max(nums[2*i],nums[2*i+1])
        return newNums

    def minMaxGame(self, nums: List[int]) -> int:
        if len(nums)==1:
            return nums[0]
        while len(nums)>1:
            nums=self.operation(nums)
        return nums[0]
        



    class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        n=len(mountain)
        liste_sortie=[]
        for i in range(1,n-1):
            if mountain[i]>mountain[i-1] and mountain[i]>mountain[i+1]:
                liste_sortie.append(i)
        return liste_sortie
    


class Solution:
    def mergeSimilarItems(self, items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
        dictionnaire=defaultdict(int)
        for liste in items1:
            dictionnaire[liste[0]]+=liste[1]
        for liste in items2:
            dictionnaire[liste[0]]+=liste[1]
        liste_sortie=[[key,val] for key,val in dictionnaire.items()]
        return sorted(liste_sortie,key=lambda x:x[0])
    


class Solution:
    def mergeArrays(self, nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
        dictionnaire=defaultdict(int)
        for liste in nums1:
            dictionnaire[liste[0]]+=liste[1]
        for liste in nums2:
            dictionnaire[liste[0]]+=liste[1]
        sortie=[[key,val] for key,val in dictionnaire.items()]
        return sorted(sortie, key=lambda x:x[0])
    


class Solution:
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        students=0
        n=len(startTime)
        for i in range(n):
            if startTime[i]<=queryTime and endTime[i]>=queryTime:
                students+=1
        return students
    


class Solution:
    def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
        dictionnaire=defaultdict(int)
        for i,liste in enumerate(mat):
            dictionnaire[i]=liste.count(1)
        max_value=max(list(dictionnaire.values()))
        liste_keys=[key for key,val in dictionnaire.items() if val==max_value]
        key=min(liste_keys)
        return [key,dictionnaire[key]]
    


class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        seen=set()
        for liste in nums:
            beginning=liste[0]
            end=liste[1]
            for i in range(beginning,end+1):
                seen.add(i)
        return len(seen)
    


class Solution:
    def wateringPlants(self, plants: List[int], capacity: int) -> int:
        current_capacity=capacity
        steps=0
        n=len(plants)
        for i in range(n):
            if current_capacity>=plants[i]:
                steps+=1
                current_capacity-=plants[i]
            else:
                steps+=2*i
                current_capacity=capacity-plants[i]
                steps+=1
        return steps
    


class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        gagnants=set()
        perdants=set()
        defeat_number=defaultdict(int)
        for match in matches:
            gagnants.add(match[0])
            perdants.add(match[1])
            perdant=match[1]
            defeat_number[perdant]+=1
        liste_gagnants=sorted([player for player in list(gagnants) if player not in perdants])
        liste_perdants=sorted([key for key,val in defeat_number.items() if val==1])
        return [liste_gagnants,liste_perdants]
    


class Solution:
    def sortEvenOdd(self, nums: List[int]) -> List[int]:
        arr=[0]*len(nums)
        even=sorted([nums[i] for i in range(0,len(nums)) if i%2==0])
        odd=sorted([nums[i] for i in range(0,len(nums)) if i%2!=0],reverse=True)
        for i in range(len(nums)):
            if i%2==0:
                arr[i]=even[i//2]
            else:
                arr[i]=odd[i//2]
        return arr
    


class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        if sorted(nums)==nums:
            return True
        if sorted(nums,reverse=True)==nums:
            return True
        return False
    

class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        frequency=defaultdict(int)
        n=len(arr)
        for num in arr:
            frequency[num]+=1/n
        for key,val in frequency.items():
            if val>0.25:
                return key
            

class Solution:
    def maximumValue(self, strs: List[str]) -> int:
        maximum_value=0
        for string in strs:
            if string.isdigit():
                maximum_value=max(maximum_value,int(string))
            else:
                maximum_value=max(maximum_value,len(string))
        return maximum_value
    


class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr.sort()
        sortie=[]
        diff_mini=float('inf')
        for i in range(len(arr)-1):
            if arr[i+1]-arr[i]<diff_mini:
                diff_mini=arr[i+1]-arr[i]
        for i in range(len(arr)-1):
            if arr[i+1]-arr[i]==diff_mini:
                sortie.append([arr[i],arr[i+1]])
        return sorted(sortie,key=lambda x:x[0])
    


class Solution:
    def countElements(self, nums: List[int]) -> int:
        occurence=defaultdict(int)
        number=0
        for num in nums:
            occurence[num]+=1
        unique=sorted(set(nums))
        for i in range(1,len(unique)-1):
            number+=occurence[unique[i]]
        return number



class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        liste_candidats=[0]*len(nums)
        for i in range(len(nums)):
            liste_candidats[i]=1-sum(nums[:i+1])
        if max(liste_candidats)>0:
            return max(liste_candidats)
        else:
            return 1
        


#Version O(n(n+k))
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        missing=[]
        i=1
        while len(missing)<k:
            if i not in arr:
                missing.append(i)
            i+=1
        return missing[k-1]
    
#Version O(n) : ce qui est demande




class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        n=len(letters)
        left=0
        right=n-1
        while left<=right:
            middle=(left+right)//2
            if letters[middle]>target:
                right=middle-1
            elif letters[middle]<=target:
                left=middle+1
        return letters[left%n]
    


class Solution:
    def threeConsecutiveOdds(self, arr: List[int]) -> bool:
        index={'even':[],'odd':[]}
        for i,num in enumerate(arr):
            if num%2==0:
                index['even'].append(i)
            else:
                index['odd'].append(i)
        for indice in index['odd']:
            if indice+1 in index['odd'] and indice+2 in index['odd']:
                return True
        return False
    


class Solution:
    def specialArray(self, nums: List[int]) -> int:
        n=len(nums)
        for i in range(n+1):
            number=len([x for x in nums if x>=i])
            if number==i:
                return i
        return -1
    


class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        population=defaultdict(int)
        for liste in logs:
            beginning,end=liste
            for i in range(beginning,end):
                population[i]+=1
        return min([key for key,val in population.items() if val==max(population.values())]) 
    



class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        count=0
        for i in range(len(hours)-1):
            for j in range(i+1,len(hours)):
                if (hours[i]+hours[j])%24==0:
                    count+=1
        return count


    

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n=len(nums)
        distinct=[0]*n
        for i in range(n):
            prefix=len(set(nums[:i+1]))
            suffix=len(set(nums[i+1:]))
            distinct[i]=prefix-suffix
        return distinct
    



class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        gagnants=0
        ballons=defaultdict(list)
        for i in range(n):
            for liste in [liste for liste in pick if liste[0]==i]:
                ballons[i].append(liste[1])
        for i in range(n):
            for element in set(ballons[i]):
                if ballons[i].count(element)>=i+1:
                    gagnants+=1
                    break
        return gagnants
    



class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        if len(text)<7:
            return 0
        dictionnaire=defaultdict(int)
        list_to_consider=[char for char in text if char in 'balloon']
        if not list_to_consider:
            return 0
        for char in list_to_consider:
            occurence_char_balloon='balloon'.count(char)
            dictionnaire[char]+=1/occurence_char_balloon
        return min([int(x) for x in list(dictionnaire.values())])
    





class Solution:
    def SmallFrequency(self, word:str)->int:
        lexique=min(word)
        return word.count(lexique)

    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
        sortie=[0]*len(queries)
        for i,quer in enumerate(queries):
            count_quer=0
            for word in words:
                if self.SmallFrequency(word)>self.SmallFrequency(quer):
                    count_quer+=1
            sortie[i]=count_quer
        return sortie
    




class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        frequency=defaultdict(int)
        words=re.findall(r'\b\w+\b', paragraph.lower())
        for word in words:
            if word not in banned:
                frequency[word]+=1
        most_common=max(frequency,key=frequency.get)
        return most_common
    




class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        indice_s=defaultdict(list)
        indice_t=defaultdict(list)
        for i,char in enumerate(s):
            indice_s[char].append(i)
        for i,char in enumerate(t):
            indice_t[char].append(i)
        liste_s=sorted(indice_s.values(),key=lambda x:x[0])
        liste_t=sorted(indice_t.values(),key=lambda x:x[0])
        return liste_s==liste_t

    

class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        liste=list(s.split())
        indice_p=defaultdict(list)
        indice_s=defaultdict(list)
        for i,char in enumerate(pattern):
            indice_p[char].append(i)
        for i,word in enumerate(liste):
            indice_s[word].append(i)
        liste1=sorted(indice_p.values(),key=lambda x:x[0])
        liste2=sorted(indice_s.values(),key=lambda x:x[0])
        return liste1==liste2
    



class Solution:
    def compatibilite(self,word1:str,word2:str)->bool:
        dict1=defaultdict(list)
        dict2=defaultdict(list)
        for i,char in enumerate(word1):
            dict1[char].append(i)
        for i,char in enumerate(word2):
            dict2[char].append(i)
        liste1=sorted(dict1.values(),key=lambda x:x[0])
        liste2=sorted(dict2.values(),key=lambda x:x[0])
        return liste1==liste2

    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        answer=[]
        for word in words:
            if self.compatibilite(word,pattern):
                answer.append(word)
        return answer
    


#Code1
class Solution:
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        seen=set()
        for liste in ranges:
            gauche,droite=liste
            for i in range(gauche,droite+1):
                seen.add(i)
        for k in range(left,right+1):
            if k not in seen:
                return False
        return True

#Code2
class Solution:
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        for num in range(left,right+1):
            is_Covered=False
            for gauche,droite in ranges:
                if gauche<=num<=droite:
                    is_Covered=True
                    break
            if not is_Covered:
                return False
        return True
    


class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n=len(nums)
        if n==1:
            return 1
        increasing=1
        decreasing=1
        max_length=1
        for i in range(1,n):
            if nums[i-1]<nums[i]:
                increasing+=1
                decreasing=1
            elif nums[i-1]>nums[i]:
                decreasing+=1
                increasing=1
            else:
                increasing=1
                decreasing=1
            max_length=max(max_length,increasing,decreasing)
        return max_length
    



class Solution:
    def closest_index(self,nums:List[int]) -> int:
        nums.sort()
        n=len(nums)
        left=0
        right=n-1
        while left<=right:
            middle=(left+right)//2
            if nums[middle]==0:
                return middle
            elif nums[middle]>0:
                right=middle-1
            else:
                left=middle+1
        if left>=n:
            return right
        if right<0:
            return left
        if abs(nums[left])<=abs(nums[right]):
            return left
        else:
            return right

    def findClosestNumber(self, nums: List[int]) -> int:
        index=self.closest_index(nums)
        return nums[index]
    



class Solution:
    def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
        number={divisor:0 for divisor in divisors}
        for divisor in set(divisors):
            for num in nums:
                if num%divisor==0:
                    number[divisor]+=1
        maximum=max(number.values())
        return min([key for key,val in number.items() if val==maximum])
    




#1ere idee naive
class Solution:
    def check(self, nums: List[int]) -> bool:
        while nums[0]!=min(nums):
            last_element=nums[-1]
            for i in range(len(nums)-1,0,-1):
                nums[i]=nums[i-1]
            nums[0]=last_element
        return nums==sorted(nums)
    

#2e idee
class Solution:
    def check(self, nums: List[int]) -> bool:
        n=len(nums)
        count=0
        for i in range(n):
            if nums[i]>nums[(i+1)%n]:
                count+=1
            if count>1:
                return False
        return True
    


class Solution:
    def checkString(self, s: str) -> bool:
        if 'a' not in s or 'b' not in s:
            return True
        index=defaultdict(list)
        for i,char in enumerate(s):
            index[char].append(i)
        return (index['a'])[-1]<(index['b'])[0]
    


class Solution:
    def areNumbersAscending(self, s: str) -> bool:
        current_num=-1
        liste=s.split()
        for chiffre in liste:
            if chiffre.isdigit():
                if int(chiffre)<=current_num:
                    return False
                else:
                    current_num=int(chiffre)
        return True
    





class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        missing_count=0
        i=0
        current=1
        while missing_count<k:
            if i<len(arr) and arr[i]==current:
                i+=1
            else:
                missing_count+=1
                if missing_count==k:
                    return current
            current+=1




class Solution:
    def isPrefixString(self, s: str, words: List[str]) -> bool:
        n=len(words)
        if words[0] not in s:
            return False
        if words[0]==s:
            return True
        for i in range(n-1):
            word=words[i]
            for j in range(i+1,n):
                word+=words[j]
                if word==s:
                    return True
        return False
    


class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        count=0
        for word in words:
            n=len(word)
            i=0
            count_word=0
            while i<min(n,len(s)) and word[i]==s[i]:
                i+=1
                count_word+=1
            if count_word==n:
                count+=1
        return count
    



class Solution:
    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        n=len(searchWord)
        index=[]
        liste=list(sentence.split())
        for i,word in enumerate(liste):
            if word[:n]==searchWord:
                index.append(i+1)
        if not index:
            return -1
        else:
            return min(index)
        


class Solution:
    def prefixCount(self, words: List[str], pref: str) -> int:
        count=0
        n=len(pref)
        for word in words:
            if word[:n]==pref:
                count+=1
        return count
    


class Solution:
    def sortSentence(self, s: str) -> str:
        liste=list(s.split())
        words=[None]*len(liste)
        for mot in liste:
            i=int(mot[-1])
            words[i-1]=mot[:-1]
        return " ".join(words)
    

#Version O(n^2)
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        if sorted(nums,reverse=True)==nums:
            return -1
        max_diff=0
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                max_diff=max(max_diff,nums[j]-nums[i])
        return max_diff
    
#Version O(n)
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        n=len(nums)
        val_min=nums[0]
        diff_max=-1
        for i in range(1,n):
            if nums[i]>val_min:
                diff_max=max(diff_max, nums[i]-val_min)
            else:
                val_min=nums[i]
        return diff_max
    


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n=len(prices)
        max_diff=0
        min_value=prices[0]
        for i in range(1,n):
            if prices[i]>min_value:
                max_diff=max(max_diff,prices[i]-min_value)
            else:
                min_value=prices[i]
        return max_diff  
    


class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        mot1=" "
        mot2=" "
        for word in word1:
            mot1+=word
        for word in word2:
            mot2+=word
        return mot1==mot2
    


class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        candidat=""
        for word in words:
            candidat+=word[0]
        return candidat==s
    

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        return -(max(nums1)-max(nums2))
    


class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr.sort()
        ecart=arr[1]-arr[0]
        n=len(arr)
        for i in range(2,n):
            if arr[i]-arr[i-1] != ecart:
                return False
        return True
    


class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        count=0
        for arr in arr1:
            drapeau=True
            for element in arr2:
                if abs(arr-element)<=d:
                    drapeau=False
                    break
            if drapeau:
                count+=1
        return count
    


class Solution:
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        ecart=defaultdict(int)
        for i,char in enumerate(keysPressed):
            if i==0:
                ecart[0]=releaseTimes[0]
            else:
                ecart[i]=releaseTimes[i]-releaseTimes[i-1]
        maximum=max(ecart.values())
        liste_keys=[key for key,val in ecart.items() if val==maximum]
        liste_candidats=[keysPressed[i] for i in liste_keys]
        return max(liste_candidats)
    


class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if min(nums)>=k:
            return 0
        return len([num for num in nums if num<k])
    


class Solution:
    def trimMean(self, arr: List[int]) -> float:
        n=len(arr)
        k=n//20
        arr.sort()
        return sum(arr[k:n-k])/(n-2*k)
    


class Solution:
    def maxDistance(self, colors: List[int]) -> int:
        n=len(colors)
        left_index_1=0
        right_index_1=n-1
        left_index_2=0
        right_index_2=n-1
        while colors[right_index_1]==colors[left_index_1]:
            right_index_1-=1
        while colors[right_index_2]==colors[left_index_2]:
            left_index_2+=1
        return max(abs(right_index_2-left_index_2),abs(right_index_1-left_index_1))
    


class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        n=len(arr)
        sortie=[0]*n
        sortie[-1]=-1
        max_value=arr[-1]
        for i in range(n-2,-1,-1):
            sortie[i]=max_value
            max_value=max(max_value,arr[i])
        return sortie
    


class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        candidates=0
        for word in words:
            liste1=[word.count(char) for char in word]
            liste2=[chars.count(char) for char in word]
            drapeau=True
            for i in range(len(liste1)):
                if liste1[i]>liste2[i]:
                    drapeau=False
                    break
            if drapeau:
                candidates+=len(word)
        return candidates
    


class Solution:
    def similar(self,s:str,t:str) -> bool:
        seen_s=set()
        seen_t=set()
        for char in s:
            seen_s.add(char)
        for char in t:
            seen_t.add(char)
        return seen_s==seen_t

    def similarPairs(self, words: List[str]) -> int:
        count=0
        for i in range(len(words)-1):
            for j in range(i+1,len(words)):
                if self.similar(words[i],words[j]):
                    count+=1
        return count
    

class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        seen=set()
        potential_judge=[]
        for liste in trust:
            seen.add(liste[0])
        for i in range(1,n+1):
            if i not in seen:
                potential_judge.append(i)
        if not potential_judge:
            return -1
        elif len(potential_judge)>1:
            return -1
        else:
            candidat=potential_judge[0]
            count=0
            for liste in trust:
                if liste[1]==candidat:
                    count+=1
            if count==n-1:
                return candidat
            else:
                return -1
            



class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        liste=list(sentence.split())
        for word in dictionary:
            n=len(word)
            for i,s in enumerate(liste):
                if s[:n]==word:
                    liste[i]=word
        return " ".join(liste)
    



class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        return len([hour for hour in hours if hour>=target])
    


class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        maximum=0
        for liste in accounts:
            maximum=max(maximum,sum(liste))
        return maximum
    


#Version O(nlog(n)) pour ne pas avoir a faire la brute
class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        nums.sort()
        n=len(nums)
        left=0
        right=n-1
        count=0
        while left<=right:
            if nums[left]+nums[right]<target:
                count+=right-left
                left+=1
            else:
                right-=1
        return count
    


class Solution:
    def leftRightDifference(self, nums: List[int]) -> List[int]:
        sortie=[0]*len(nums)
        sortie[0]=sum(nums[1:])
        sortie[-1]=sum(nums[:len(nums)-1])
        for i in range(1,len(nums)-1):
            sortie[i]=abs(sum(nums[:i]) - sum(nums[i+1:]))
        return sortie
    


class Solution:
    def mostWordsFound(self, sentences: List[str]) -> int:
        maximum=0
        for sentence in sentences:
            taille=len(list(sentence.split()))
            maximum=max(maximum,taille)
        return maximum
    


class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        n=len(A)
        sortie=[0]*n
        for i in range(n):
            j=0
            while j<=i:
                if A[j] in B[:i+1]:
                    sortie[i]+=1
                j+=1
        return sortie
    



class Solution:
    def numOfPairs(self, nums: List[str], target: str) -> int:
        count=0
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i!=j and nums[i]+nums[j]==target:
                    count+=1
        return count
    


class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        n=len(boxes)
        answer=[0]*n
        for i in range(n):
            for j in range(n):
                if j!=i and boxes[j]=='1':
                    answer[i]+=abs(i-j)
        return answer
    



class Solution:
    def greatestLetter(self, s: str) -> str:
        majuscule=defaultdict(int)
        minuscule=defaultdict(int)
        for char in s:
            if char.isupper():
                majuscule[char]+=1
            else:
                minuscule[char]+=1
        candidats=[key for key in minuscule.keys() if key.upper() in majuscule.keys()]
        candidats=sorted(candidats,reverse=True)
        if candidats:
            return candidats[0].upper()
        else:
            return ""
        



class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        letters='abcdefghijklmnopqrstuvwxyz'
        for letter in letters:
            if letter not in sentence:
                return False
        return True
    



class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        count=0
        liste=[list(set(word)) for word in words]
        for word in liste:
            for char in word:
                if char not in allowed:
                    count+=1
                    break
        return len(words)-count
    


class Solution:
    def countPoints(self, rings: str) -> int:
        taille=len(rings)
        count=0
        couleurs=defaultdict(list)
        for i in range(0,taille-1,2):
            couleurs[rings[i+1]].append(rings[i])
        for key in couleurs.keys():
            if len(set(couleurs[key]))==3:
                count+=1
        return count
    


class Solution:
    def numberOfSpecialChars(self, word:str) -> int:
        word=set(word)
        minuscule=defaultdict(int)
        majuscule=defaultdict(int)
        for char in word:
            if char.isupper():
                majuscule[char]+=1
            else:
                minuscule[char]+=1
        candidats=[key for key in minuscule.keys() if key.upper() in majuscule.keys()]
        return len(candidats)
    

#Version 1
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if word.upper()==word:
            return True
        if word.lower()==word:
            return True
        if not word[0].isupper():
            return False
        for i in range(1,len(word)):
            if word[i].isupper():
                return False
        return True
    

#Version2
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if word.upper()==word:
            return True
        if word.lower()==word:
            return True
        if word.istitle():
            return True
        else:
            return False
        


class Solution:
    def capitalizeTitle(self, title: str) -> str:
        candidats=list(title.split())
        new=[]
        for word in candidats:
            if len(word) in [1,2]:
                new.append(word.lower())
            else:
                new.append(word.title())
        return ' '.join(new)
    


class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        count=0
        minuscule=defaultdict(list)
        majuscule=defaultdict(list)
        for i,char in enumerate(word):
            if char.isupper():
                majuscule[char].append(i)
            else:
                minuscule[char].append(i)
        candidats=[key for key in minuscule.keys() if key.upper() in majuscule.keys()]
        for candidat in candidats:
            if max(minuscule[candidat])<min(majuscule[candidat.upper()]):
                count+=1
        return count
    


class Solution:
    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        count=0
        for item in items:
            types,color,name=item
            if ruleKey=="type" and types==ruleValue:
                count+=1
            elif ruleKey=="color" and color==ruleValue:
                count+=1
            elif ruleKey=="name" and name==ruleValue:
                count+=1
        return count
    


class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        num1=sum([x for x in range(1,n+1) if x%m!=0])
        num2=sum([x for x in range(1,n+1) if x%m==0])
        return num1-num2
    
#Probleme tagué comme recherche binaire mais pas naturel pour moi. On peut faire bcp mieux en terme de temps par contre
class Solution:
    def arrangeCoins(self, n: int) -> int:
        i=1
        count=0
        while n-i>=0:
            count+=1
            n-=i
            i+=1
        return count
    


class Solution:
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        count=0
        n=len(nums)
        for i in range(n-2):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    if nums[j]-nums[i]==diff and nums[k]-nums[j]==diff:
                        count+=1
        return count
    


class Solution:
    def unequalTriplets(self, nums: List[int]) -> int:
        n=len(nums)
        count=0
        for i in range(n-2):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    if nums[i]!=nums[j] and nums[i]!=nums[k] and nums[j]!=nums[k]:
                        count+=1
        return count
    


class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        maximum=0
        n=len(nums)
        for i in range(n-2):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    candidat=(nums[i]-nums[j])*nums[k]
                    maximum=max(maximum,candidat)
        return maximum
    


class Solution:
    def findingUsersActiveMinutes(self, logs: List[List[int]], k: int) -> List[int]:
        answer=[0]*k
        dictionnaire=defaultdict(set)
        for liste in logs:
            identite,minute=liste
            dictionnaire[identite].add(minute)
        uam={i:0 for i in range(1,k+1)}
        for val in dictionnaire.values():
            uam[len(val)]+=1
        for i in range(k):
            answer[i]=uam[i+1]
        return answer
    

class Solution:
    def removeStars(self, s: str) -> str:
        stack=[]
        for char in s:
            if char!='*':
                stack.append(char)
            else:
                stack.pop()
        return ''.join(stack)
    


class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack=[]
        for char in s:
            if stack and char==stack[-1]:
                stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)
    


class Solution:
    def makeGood(self, s: str) -> str:
        stack=[]
        for char in s:
            if stack and char!=stack[-1] and char.upper()==stack[-1]:
                stack.pop()
            elif stack and char!=stack[-1] and char.lower()==stack[-1]:
                stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)
    


class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        stack=[]
        for char in s:
            if len(stack)>=k-1 and stack[-(k-1):]==[char]*(k-1):
                stack=stack[:-(k-1)]
            else:
                stack.append(char)
        return ''.join(stack)
    


class Solution:
    def checkDistances(self, s: str, distance: List[int]) -> bool:
        n=len(distance)
        index=defaultdict(list)
        for i,char in enumerate(s):
            index[char].append(i)
        for char in s:
            indice=ord(char)-ord('a')
            if distance[indice]!=index[char][1]-index[char][0]-1:
                return False
        return True
    


class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
        n=len(s)
        answer=[0]*n
        index=defaultdict(list)
        for i,char in enumerate(s):
            index[char].append(i)
        liste=index[c]
        for i in range(n):
            answer[i]=min([abs(i-element) for element in liste])
        return answer
    

#Marche pas sur des cas grands mais formule combinatoire trop difficile
 class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a=[1]*n
        compteur=0
        while compteur<k:
            for i in range(1,n):
                a[i]=a[i]+a[i-1]
            compteur+=1
        return a[n-1]

    

class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five_dollars=0
        ten_dollars=0
        for bill in bills:
            if bill==5:
                five_dollars+=1
            elif bill==10:
                if five_dollars==0:
                    return False
                else:
                    ten_dollars+=1
                    five_dollars-=1
            elif bill==20:
                if ten_dollars>0 and five_dollars>=1:
                    ten_dollars-=1
                    five_dollars-=1
                elif five_dollars>=3:
                    five_dollars-=3
                else:
                    return False
        return True
    


class Solution:
    def stringMatching(self, words: List[str]) -> List[str]:
        n=len(words)
        answer=[]
        words=sorted(words,key=lambda x:len(x))
        for i,word in enumerate(words):
            for j in range(i+1,n):
                if word in words[j]:
                    answer.append(word)
                    break
        return answer
    



class Solution:
    def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
        count=0
        current_sum=0
        seen=set(banned)
        for i in range(1,n+1):
            if i not in seen:
                if current_sum+i<=maxSum:
                    count+=1
                    current_sum+=i
                else:
                    break
        return count
    


class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        seen=set(nums)
        i=1
        while i in seen:
            i+=1
        return i
    

class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        return nums + nums
    

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        n=len(nums)
        nums.sort()
        array=[]
        alice=[nums[i] for i in range(n) if i%2==0]
        bob=[nums[i] for i in range(n) if i%2!=0]
        k=n//2
        for j in range(k):
            array.append(bob[j])
            array.append(alice[j])
        return array
    

class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:
        n=len(nums)
        array=[0]*n
        for i,element in enumerate(nums):
            array[i]=nums[element]
        return array
    


class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        liste1=nums[:n]
        liste2=nums[n:]
        answer=[]
        for i in range(n):
            answer.append(liste1[i])
            answer.append(liste2[i])
        return answer
    


class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        n=len(candies)
        maximum=max(candies)
        result=[False]*n
        for i,candy in enumerate(candies):
            if candy+extraCandies>=maximum:
                result[i]=True
            else:
                result[i]=False
        return result
    


class Solution:
    def countDigits(self, num: int) -> int:
        liste=[char for char in str(num) if num%int(char)==0]
        return len(liste)
    


class Solution:
    def differenceOfSum(self, nums: List[int]) -> int:
        element=sum(nums)
        absolute=0
        for num in nums:
            for char in str(num):
                absolute+=int(char)
        return abs(element-absolute)
    


class Solution:
    def alternateDigitSum(self, n: int) -> int:
        somme=0
        liste=[int(char) for char in str(n)]
        produit=1
        for element in liste:
            somme+=element*produit
            produit*=-1
        return somme
    

class Solution:
    def separateDigits(self, nums: List[int]) -> List[int]:
        answer=[]
        for num in nums:
            for char in str(num):
                answer.append(int(char))
        return answer
    


class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        n=len(coordinates)
        if n==2:
            return True
        x0,y0=coordinates[0]
        x1,y1=coordinates[1]
        initial_dx=x1-x0
        initial_dy=y1-y0
        i=2
        while i<=n-1:
            xi,yi=coordinates[i]
            ai,bi=coordinates[i-1]
            if (yi-bi)*initial_dx != (xi-ai)*initial_dy:
                return False
            else:
                i+=1
        return i==n
    


class Solution:
    def formation(self,nums:List[int]) -> bool:
        n=len(nums)
        for i in range(n):
            l=sum([nums[j] for j in range(n) if j!=i])
            if l<=nums[i]:
                return False
        return True

    def triangleType(self, nums: List[int]) -> str:
        nums.sort()
        if len(set(nums))==1:
            return "equilateral"
        if not self.formation(nums):
            return "none"
        else:
            if len(set(nums))==2:
                return "isosceles"
            else:
                return "scalene"
            


class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        groupes=defaultdict(int)
        for card in deck:
            groupes[card]+=1
        if min(groupes.values())<=1:
            return False
        pgcd=reduce(math.gcd,list(groupes.values()))
        return pgcd>1
    


class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        answer=[]
        letters=[char for char in s if char.isalpha()]
        for char in s:
            if char.isalpha():
                answer.append(letters.pop())
            else:
                answer.append(char)
        return ''.join(answer)
    



class Solution:
    def getLucky(self, s: str, k: int) -> int:
        string=''
        for char in s:
            position=str(ord(char)-ord('a')+1)
            string=string+position
        j=1
        while j<=k:
            x=sum([int(char) for char in string])
            string=str(x)
            j+=1
        return x
    


class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        sentence=list(sentence.split())
        n=len(sentence)
        if n==1:
            return sentence[0][0]==sentence[0][-1]
        if sentence[-1][-1]!=sentence[0][0]:
            return False
        for i in range(n-1):
            if sentence[i][-1]!=sentence[i+1][0]:
                return False
        return True
    



class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        text=list(text.split())
        n=len(text)
        answer=[]
        for i in range(n-2):
            if text[i]==first and text[i+1]==second:
                answer.append(text[i+2])
        return answer



class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s=list(s.split())
        return len(s[-1])
    

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle not in haystack:
            return -1
        n=len(haystack)
        m=len(needle)
        for i in range(n-m+1):
            if haystack[i:i+m]==needle:
                return i
            

#version1: peu efficace et TLE 
class Solution:
    def oneShift(self, s:str) -> str:
        liste=[char for char in s]
        n=len(liste)
        string=[0]*n
        for i in range(n-1):
            string[i]=liste[i+1]
        string[-1]=liste[0]
        return ''.join(string)

    def rotateString(self, s: str, goal: str) -> bool:
        while s[0]!=goal[0]:
            goal=self.oneShift(goal)
        return s==goal
    

#Version2: plus astucieuse
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        if len(s)!=len(goal):
            return False
        s=s+s
        return (goal in s)
    

class Solution:
    def minimumMoves(self, s: str) -> int:
        if 'X' not in s:
            return 0
        n=len(s)
        number=0
        i=0
        while i<n:
            if s[i]=='X':
                number+=1
                i+=3
            else:
                i+=1
        return number
    


class Solution:
    def thousandSeparator(self, n: int) -> str:
        if len(str(n))<=3:
            return str(n)
        string=list(reversed(str(n)))
        taille=len(string)
        answer=[string[0]]
        for i in range(1,taille):
            if i%3==0:
                answer.append('.')
                answer.append(string[i])
            else:
                answer.append(string[i])
        answer.reverse()
        return ''.join(answer)
    



class Solution:
    def reformat(self, s: str) -> str:
        if len(s)==1:
            return s
        if len([char for char in s if char.isdigit()]) in [0,len(s)]:
            return ""
        answer=[]
        produit=1
        liste2=list(reversed([char for char in s if char.isdigit()]))
        liste1=list(reversed([char for char in s if char.isalpha()]))
        if abs(len(liste1)-len(liste2))>=2:
            return ""
        if len(liste2)>len(liste1):
            produit=-1
        while len(answer)<len(s):
            if produit==1:
                if liste1:
                    answer.append(liste1.pop())
            else:
                if liste2:
                    answer.append(liste2.pop())
            produit*=-1
        return ''.join(answer)



#Version1
class Solution:
    def makeFancyString(self, s: str) -> str:
        stack=[]
        for char in s:
            if len(stack)>=2 and stack[-1]==stack[-2]==char:
                stack.append(char)
                stack.pop() 
            else:
                stack.append(char)
        return ''.join(stack)
    

#Version2
class Solution:
    def makeFancyString(self, s: str) -> str:
        stack=[]
        for char in s:
            if len(stack)>=2 and stack[-1]==stack[-2]==char:
                continue
            else:
                stack.append(char)
        return ''.join(stack)
    


class Solution:
    def isValid(self, word: str) -> bool:
        vowels='aeiouAEIOU'
        if len(word)<3:
            return False
        vowels_number=len([char for char in word if char in vowels])
        conso_number=len([char for char in word if char.isalpha() and char not in vowels])
        if vowels_number==0:
            return False
        if conso_number==0:
            return False
        for char in word:
            if not char.isdigit() and not char.isalpha():
                return False
        return True
    



class Solution:
    def convertTime(self, current: str, correct: str) -> int:
        operations=0
        temps2=int(correct[:2])*60+int(correct[3:])
        temps1=int(current[:2])*60+int(current[3:])
        ecart=temps2-temps1
        liste=[60,15,5,1]
        i=0
        while i<len(liste):
            x=ecart//liste[i]
            ecart-=x*liste[i]
            operations+=x
            i+=1
        return operations
    



class Solution:
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        index=defaultdict(list)
        for i,char in enumerate(s):
            index[char].append(i)
        if max([len(val) for val in index.values()])<2:
            return -1
        maximum=0
        for val in index.values():
            maximum=max(maximum,max(val)-min(val)-1)
        return maximum
    



class Solution:
    def largestGoodInteger(self, num: str) -> str:
        stack=[]
        answer=[]
        for char in num:
            if len(stack)>=2 and stack[-1]==stack[-2]==char:
                answer.append(char*3)
            else:
                stack.append(char)
        if len(answer)==0:
            return ""
        return sorted(answer)[-1]
    


class Solution:
    def scoreOfString(self, s: str) -> int:
        n=len(s)
        somme=0
        for i in range(n-1):
            somme+=abs(ord(s[i+1])-ord(s[i]))
        return somme
    



class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        if ch not in word:
            return word
        index=0
        for i,char in enumerate(word):
            if char==ch:
                index=i
                break
        mot=''.join(list(reversed(word[:i+1])))
        return mot+word[i+1:]
    


class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        index=defaultdict(str)
        answer=['']*len(s)
        for i in range(len(s)):
            answer[indices[i]]=s[i]
        return ''.join(answer)
    

class Solution:
    def deleteGreatestValue(self, grid: List[List[int]]) -> int:
        answer=0
        m=len(grid[0])
        while m>0:
            maximum=0
            for liste in grid:
                liste.sort()
                if liste:
                    maximum=max(liste.pop(),maximum)
            answer+=maximum
            m-=1
        return answer
    


class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        candidats=[min(liste) for liste in matrix]
        confirmes=[]
        dic=defaultdict(list)
        for ligne in matrix:
            for i,element in enumerate(ligne):
                dic[i].append(element)
        for i,candidat in enumerate(candidats):
            index=matrix[i].index(candidat)
            if candidat==max(dic[index]):
                confirmes.append(candidat)
        return confirmes
    



class Solution:
    def reverse(self, nums:List[int]) -> List[int]:
        return list(reversed(nums))
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        matrix=[]
        for liste in image:
            liste=list(reversed(liste))
            for i,element in enumerate(liste):
                if element==0:
                    liste[i]=1
                else:
                    liste[i]=0
            matrix.append(liste)
        return matrix



class Solution:
    def firstPalindrome(self, words: List[str]) -> str:
        for word in words:
            if word==word[::-1]:
                return word
        return ""
    


class Solution:
    def reverseWords(self, s: str) -> str:
        answer=[]
        s=list(s.split())
        for word in s:
            answer.append(word[::-1])
        return ' '.join(answer)


#Methode 1
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        s=list(s)
        n=len(s)
        for i in range(0,n,2*k):
            s[i:i+k]=reversed(s[i:i+k])
        return ''.join(s)
    
#Methode 2:two pointers
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        s=list(s)
        n=len(s)
        for i in range(0,n,2*k):
            left,right=i,min(i+k-1,n-1)
            while left<=right:
                s[left],s[right]=s[right],s[left]
                left+=1
                right-=1
        return ''.join(s)
    


class Solution:
    def Palindrome(self,s:str) ->str:
        s=list(s)
        return ''.join(reversed(s))

    def isPalindrome(self, s: str) -> bool:
        if s=="":
            return True
        liste=[char.lower() for char in s if char.isalnum()]
        string=''.join(liste)
        return string==self.Palindrome(string)



class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        liste=[]
        m=min(len(word1),len(word2))
        pointer=0
        while pointer<m:
            letter1=word1[pointer]
            letter2=word2[pointer]
            liste.append(letter1)
            liste.append(letter2)
            pointer+=1
        return ''.join(liste) + word1[m:] + word2[m:]
    


class Solution:
    def reverseString(self, s: List[str]) -> None:
        """   
        Do not return anything, modify s in-place instead.
        """
        left=0
        right=len(s)-1
        while left<=right:
            s[left],s[right]=s[right],s[left]
            left+=1
            right-=1



class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        i=0
        j=i+1
        while i<len(nums)-1:
            if nums[i]==nums[j]:
                nums[i]*=2
                nums[j]=0
            i+=1
            j+=1
        return [x for x in nums if x!=0] + [x for x in nums if x==0]
 

class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        answer=[]
        positive=[num for num in nums if num>0]
        negative=[num for num in nums if num<0]
        i=0
        while i<len(positive):
            answer.append(positive[i])
            answer.append(negative[i])
            i+=1
        return answer
    


class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        result=[0]*len(nums)
        for i in range(len(nums)):
            result[i]=sum(nums[:i+1])
        return result
    


class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        vowels='aeiouAEIOU'
        number1=0
        number2=0
        for char in s[:len(s)//2]:
            if char in vowels:
                number1+=1
        for char in s[len(s)//2:]:
            if char in vowels:
                number2+=1
        return number1==number2
    



class Solution:
    def sortVowels(self, s: str) -> str:
        vowels='aeiouAEIOU'
        s=list(s)
        candidats=sorted([char for char in s if char in vowels])
        j=0
        for i in range(len(s)):
            if s[i] in vowels:
                s[i]=candidats[j]
                j+=1
        return ''.join(s)
    


class Solution:
    def arrangeWords(self, text: str) -> str:
        text=list(text.split())
        text=sorted(text, key=lambda x:len(x))
        text[0]=text[0].title()
        for i in range(1,len(text)):
            text[i]=text[i].lower()
        return ' '.join(text)
    

#consigne non respectee
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n0=nums.count(0)
        n1=nums.count(1)
        n2=len(nums)-n1-n0
        for i in range(n0):
            nums[i]=0
        for i in range(n0,n0+n1):
            nums[i]=1
        for i in range(n0+n1,len(nums)):
            nums[i]=2



class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        number=0
        i=0
        while i<len(costs) and coins-costs[i]>=0:
            number+=1
            coins-=costs[i]
            i+=1
        return number
    

class Solution:
    def power(self, n:int) -> int:
        power=0
        while n!=1:
            if n%2==0:
                n=n//2
            else:
                n=3*n+1
            power+=1
        return power

    def getKth(self, lo: int, hi: int, k: int) -> int:
        liste=sorted([x for x in range(lo,hi+1)],key=lambda x:self.power(x))
        return liste[k-1]
    


class Solution:
    def critere(self, word:str) -> bool:
        word=list(word)
        vowels='aeiou'
        if word[0] in vowels and word[-1] in vowels:
            return True
        return False

    def vowelStrings(self, words: List[str], left: int, right: int) -> int:
        number=0
        for i in range(left,right+1):
            if self.critere(words[i]):
                number+=1
        return number
    


class Solution:
    def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:
        string1=''
        string2=''
        string3=''
        for char in firstWord:
            string1+=str(ord(char)-ord('a'))
        for char in secondWord:
            string2+=str(ord(char)-ord('a'))
        for char in targetWord:
            string3+=str(ord(char)-ord('a'))
        return int(string1) + int(string2) == int(string3)
    

#Version1
class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels='aeiouAEIOU'
        s=list(s)
        voyelle=defaultdict(str)
        for i,char in enumerate(s):
            if char in vowels:
                voyelle[i]=char
        reversed_vowels=list(reversed(list(voyelle.values())))
        for index,val in zip(voyelle.keys(),reversed_vowels):
            s[index]=val
        return ''.join(s)
    

#Version 2: TRES TRES elegante
class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels='aeiouAEIOU'
        s=list(s)
        left=0
        right=len(s)-1
        while left<=right:
            if s[left] not in vowels:
                left+=1
            elif s[right] not in vowels:
                right-=1
            else:
                s[left],s[right]=s[right],s[left]
                left+=1
                right-=1
        return ''.join(s)
    


class Solution:
    def findIndices(self, nums: List[int], indexDifference: int, valueDifference: int) -> List[int]:
        n=len(nums)
        for left in range(n):
            right=n-1
            while left<=right:
                if abs(left-right)>=indexDifference and abs(nums[left]-nums[right])>=valueDifference:
                    return [left,right]
                right-=1
        return [-1,-1]
    


class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        petits=[x for x in nums if x<pivot]
        grands=[x for x in nums if x>pivot]
        return petits + [pivot]*nums.count(pivot) + grands
    

class Solution:
    def modifyString(self, s: str) -> str:
        s=list(s)
        alphabet='abcdefghijklmnopqrstuvwxyz'
        if len(s)==1 and s[0]=='?':
            return "a"
        for i,char in enumerate(s):
            if char=='?':
                if i==0:
                    s[i]=random.choice([c for c in alphabet if c!=s[i+1]])
                elif i==len(s)-1:
                    s[i]=random.choice([c for c in alphabet if c!=s[i-1]])
                else:
                    s[i]=random.choice([c for c in alphabet if c!=s[i+1] and c!=s[i-1]])
        return ''.join(s)
    


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs)==1:
            return strs[0]
        strs=sorted(strs,key=lambda x:len(x))
        candidat=list(strs[0])
        answer=[]
        for i in range(len(candidat)):
            if len(set([mot[i] for mot in strs]))==1:
                answer.append(candidat[i])
            else:
                break
        return ''.join(answer)
    

class Solution:
    def customSortString(self, order: str, s: str) -> str:
        s=list(s)
        o=list(order)
        dictionnaire={i:char for i,char in enumerate(s) if char in order}
        order_characters=sorted(list(dictionnaire.values()), key=lambda x:o.index(x))
        for i,val in zip(dictionnaire.keys(), order_characters):
            s[i]=val
        return ''.join(s)
    

class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        candidat=list(words[0])
        answer=[]
        for char in candidat:
            drapeau=True
            for i in range(1,len(words)):
                if char not in words[i]:
                    drapeau=False
                    break
                else:
                    words[i]=words[i].replace(char,'',1)
            if drapeau:
                answer.append(char)
        return answer
    

#Code absolument HORRIBLE mais marche a priori pour un pb avec 50% d'acceptation
class Solution:
    def checkValid(self, matrix: List[List[int]]) -> bool:
        n=len(matrix)
        for liste in matrix:
            for i in range(1,n+1):
                if i not in liste:
                    return False
        colonnes=defaultdict(list)
        for i in range(n):
            for liste in matrix:
                colonnes[i].append(liste[i])
        for i in range(1,n+1):
            for val in colonnes.values():
                if i not in val:
                    return False
        return True
    

#Contraintes non respectees
class Solution:
    def operation(self, s:str) -> str:
        stack=[]
        for char in s:
            if char!='#':
                stack.append(char)
            else:
                if stack:
                    stack.pop()
        return ''.join(stack)

    def backspaceCompare(self, s: str, t: str) -> bool:
        return self.operation(s)==self.operation(t)
    

class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        s=list(s)
        index=[]
        i=0
        while i<len(s):
            j=i 
            while j<len(s) and s[j]==s[i]:
                j+=1
            if j-i>=3:
                index.append([i,j-1])
            i=j
        return index
    

class Solution:
    def result(self, num:int, divisor:int) -> int:
        if num%divisor==0:
            return num//divisor
        else:
            return num//divisor + 1

    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        left=1
        right=max(nums)
        while left<right:
            middle=(left+right)//2
            somme=somme=sum([self.result(num,middle) for num in nums])
            if somme>threshold:
                left=middle+1
            else:
                right=middle
        return left
    

class Solution:
    def clearDigits(self, s: str) -> str:
        stack=[]
        for char in s:
            if char.isdigit():
                if stack:
                    stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)
    

class Solution:
    def minOperations(self, logs: List[str]) -> int:
        number=0
        for string in logs:
            if string=="../":
                if number>0:
                    number-=1
            elif string!="./":
                number+=1
        return number
    

class Solution:
    def longestPalindrome(self, s: str) -> int:
        number=0
        impair=False
        occurence=defaultdict(int)
        for char in s:
            occurence[char]+=1
        for val in occurence.values():
            if val%2==0:
                number+=val
            else:
                number+=val-1
                impair=True
        if impair:
            number+=1
        return number
    

#Code HORRIBLE
class Solution:
    def operation(self, string:str) -> List[int]:
        return [ord(char)-ord('a') for char in string]

    def oddString(self, words: List[str]) -> str:
        n=len(words)
        difference=[]
        for string in words:
            liste=self.operation(string)
            diff=[liste[j+1]-liste[j] for j in range(len(liste)-1)]
            difference.append(diff)
        reference=difference[0]
        count=[]
        for i in range(1,n):
            if difference[i]==reference:
                count.append(i)
        if count==[]:
            return words[0]
        index=[i for i in range(1,n) if i not in count][0]
        return words[index]
    

class Solution:
    def distinctAverages(self, nums: List[int]) -> int:
        seen=set()
        nums.sort()
        while len(nums)>=2:
            maximum=nums.pop()
            minimum=nums.pop(0)
            moyenne=(minimum+maximum)/2
            seen.add(moyenne)
        return len(seen)
    

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        somme=0
        for i in range(len(nums)):
            index=i+1
            if len(nums)%index==0:
                somme+=nums[i]**2
        return somme
    
class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        for i in range(n):
            seen = set()  
            for j in range(i, n):
                seen.add(nums[j])  
                count_distinct = len(seen)
                total_sum += count_distinct **2
        return total_sum
    


class Solution:
    def isNice(self, s:str) -> bool:
        for char in set(s):
            if char.isupper():
                if char.lower() not in s:
                    return False
            else:
                if char.upper() not in s:
                    return False
        return True

    def longestNiceSubstring(self, s: str) -> str:
        longest=""
        n=len(s)
        for i in range(n-1):
            for j in range(i+1,n):
                substring=s[i:j+1]
                if self.isNice(substring) and len(substring)>len(longest):
                    longest=substring
        return longest


    
class Solution:
    def reverse(self, s:str) -> str:
        return s[::-1]

    def reverseParentheses(self, s: str) -> str:
        stack=[]
        s=list(s)
        for i,char in enumerate(s):
            if char=='(':
                stack.append(i)
            elif char==')':
                j=stack.pop()
                s[j+1:i]=self.reverse(s[j+1:i])
        return ''.join([char for char in s if char not in '()'])
    


class Solution:
    def decodeString(self, s: str) -> str:
        stack=[]
        s=list(s)
        for i,char in enumerate(s):
            if char=='[':
                stack.append(i)
            elif char==']':
                j=stack.pop()
                s[j-1:i+1]=int(s[j-1])*s[j+1:i]
        if '[' not in s:
            return ''.join([char for char in s])
        else:
            i=s.index('[')
            j=s.index(']')
            s[i-1:j+1]=int(s[i-1])*s[i+1:j]
            return ''.join([char for char in s])
        


class Solution:
    def complexNumberMultiply(self, num1: str, num2: str) -> str:
        indice1=num1.index('+')
        indice2=num2.index('+')
        reel1=int(num1[:indice1])
        reel2=int(num2[:indice2])
        im1=int(num1[indice1+1:-1])
        im2=int(num2[indice2+1:-1])
        x=reel1*reel2 - im1*im2
        y=reel1*im2 + reel2*im1
        return f"{x}+{y}i"
    


#Aide chat GPT
class Solution:
    def minimumDeletions(self, s: str) -> int:
        number=0
        count=0
        for char in s:
            if char=='b':
                number+=1
            elif char=='a':
                if number>0:
                    number-=1
                    count+=1
        return count
    


#1e version a reprendre
class Solution:
    def rle(self, s:str)->str:
        occurence=defaultdict(int)
        answer=""
        for char in s:
            occurence[char]+=1
        for char in set(s):
            answer+=str(occurence[char])
            answer+=char
        return answer

    def countAndSay(self, n: int) -> str:
        if n==1:
            return "1"
        else:
            answer='1'
            for _ in range(2,n+1):
                answer=self.rle(answer)
            return answer
        

#1e version naive
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        answer=[]
        n=len(s)
        for i in range(n):
            for j in range(i+1,n+1):
                substring=s[i:j]
                if substring==substring[::-1] and substring!="":
                    answer.append(substring)
        return answer
    


class Solution:
    def reverseWords(self, s: str) -> str:
        words=s.split()
        words=words[::-1]
        return ' '.join(words)
    

#1ere methode: tres couteuse en terme de conversion integer -> string
class Solution:
    def factorial(self,n:int) -> int:
        x=1
        for i in range(2,n+1):
            x=x*i
        return x

    def trailingZeroes(self, n: int) -> int:
        if n==0:
            return 0
        number=str(self.factorial(n))
        count=0
        i=len(number)-1
        while i>=0 and number[i]=='0':
            count+=1
            i-=1
        return count
    
#un trailing zero est fait avec un facteur 10 : 2 et 5 mais comme il apparait toujours un 2 dans les facteurs pairs,
#il suffit de compter le nombre de multiples de 5
    class Solution:
    def trailingZeroes(self, n: int) -> int:
        if n==0:
            return 0
        count=0
        while n>0:
            n//=5
            count+=n
        return count
    


class Solution:
    def reorderSpaces(self, text: str) -> str:
        words=text.split()
        spaces=text.count(' ')
        m=len(words)
        if m==1:
            return ' '.join(words) + ' '*spaces
        distribution=spaces//(m-1)
        fin=spaces%(m-1)
        result=(' '*distribution).join(words)
        result+=' '*fin
        return result
    

#926/1306
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        letters=defaultdict(int)
        for char in word1:
            letters[char]+=1
        for char in word2:
            letters[char]-=1
        return sum([abs(val) for val in letters.values() if val!=0])
    



class Solution:
    def shift(self,s:str,n:str) ->str:
        word='abcdefghijklmnopqrstuvwxyz'
        indice1=word.index(s)
        return word[indice1+int(n)]

    def replaceDigits(self, s: str) -> str:
        s=list(s)
        n=len(s)
        answer=['']*n
        for i in range(n):
            if i%2==0:
                answer[i]=s[i]
            else:
                letter=self.shift(s[i-1],s[i])
                answer[i]=letter
        return ''.join(answer)
    



class Solution:
    def shift(self,s:str,n:int):
        mot='abcdefghijklmnopqrstuvwxyz'
        indice=mot.index(s)
        return mot[(indice+n)%26]

    def shiftingLetters(self, s: str, shifts: List[int]) -> str:
        s=list(s)
        n=len(s)
        rotation=0
        for i in range(n-1,-1,-1):
            rotation+=shifts[i]
            ajout=rotation%26
            s[i]=self.shift(s[i],ajout)
        return ''.join(s)
    


class Solution:
    def numOfStrings(self, patterns: List[str], word: str) -> int:
        count=0
        for mot in patterns:
            if mot in word:
                count+=1
        return count
    


class Solution:
    def makeEqual(self, words: List[str]) -> bool:
        occurence=defaultdict(int)
        m=len(words)
        if m==1:
            return True
        for mot in words:
            for char in mot:
                occurence[char]+=1
        for val in list(occurence.values()):
            if val%m!=0:
                return False
        return True
    



class Solution:
    def nextLetter(self, s:str) -> str:
        word='abcdefghijklmnopqrstuvwxyz'
        indice=word.index(s)
        return word[(indice+1)%26]

    def operation(self, s:str) -> str:
        n=len(s)
        for i in range(n):
            s+=self.nextLetter(s[i])
        return s
            
    def kthCharacter(self, k: int) -> str:
        word='a'
        while len(word)<k:
            word=self.operation(word)
        return word[k-1]



class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        answer=[]
        m=len(matrix)
        n=len(matrix[0])
        for i in range(n):
            colonne=[]
            for j in range(m):
                colonne.append(matrix[j][i])
            answer.append(colonne)
        return answer
    

#TLE
class Solution:
    def isAnagram(self, s:str, p:str) -> bool:
        occurence=defaultdict(int)
        for char in s:
            occurence[char]+=1
        for char in p:
            occurence[char]-=1
        return max(list(occurence.values()))==0

    def findAnagrams(self, s: str, p: str) -> List[int]:
        indices=[]
        n=len(s)
        m=len(p)
        i=0
        while i<=n-m:
            if self.isAnagram(s[i:i+m],p):
                indices.append(i)
            i+=1
        return indices
    


class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        stack=[0]
        for char in s:
            if char=='(':
                stack.append(0)
            elif char==')':
                v=stack.pop()
                stack[-1]+=max(2*v,1)
        return stack[-1]
    


class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        n=len(values)
        max_left=values[0]
        maximum=0
        for i in range(1,n):
            maximum=max(maximum, max_left+values[i]-i)
            max_left=max(values[i]+i,max_left)
        return maximum
    


class Solution:
    def letter(self, s:str) -> str:
        n=int(s)
        return chr(n + 96)

    def freqAlphabets(self, s: str) -> str:
        answer=[]
        mot=list(s)
        i=len(s)-1
        while i>=0:
            if mot[i]=='#':
                answer.append(self.letter(s[i-2:i]))
                i-=3
            else:
                answer.append(self.letter(s[i]))
                i-=1
        return ''.join(reversed(answer))
    


class Solution:
    def maximum69Number (self, num: int) -> int:
        s=str(num)
        mot=list(s)
        i=0
        while i<len(s):
            if int(mot[i])==6:
                mot[i]='9'
                return int(''.join(mot))
            i+=1
        return num
    

#78/80
class Solution:
    def removeOccurrences(self, s: str, part: str) -> str:
        m=len(part)
        while part in s:
            i=0
            while i<len(s)-m+1:
                if s[i:i+m]==part:
                    s=s[0:i]+s[i+m:]
                i+=1
        return s
    



class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        somme=0
        seen=set()
        n=len(mat)
        for i in range(n):
            somme+=mat[i][i]
            seen.add((i,i))
        i=0
        while i<=n-1:
            if (i,n-1-i) not in seen:
                somme+=mat[i][n-1-i]
            i+=1
        return somme
    


class Solution:
    def checkXMatrix(self, grid: List[List[int]]) -> bool:
        n=len(grid)
        for i in range(n):
            if grid[i][i]==0 or grid[i][n-1-i]==0:
                return False
            for j in range(n):
                if j!=i and j!=n-1-i:
                    if grid[i][j]!=0:
                        return False
        return True
    


class Solution:
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        answer=[]
        m=len(score) #lignes
        n=len(score[0]) #colonnes
        resultat=defaultdict(int)
        for i in range(m):
            resultat[i]=score[i][k]
        liste=sorted(resultat.keys(), key=lambda x: -resultat[x])
        for j in liste:
            answer.append(score[j])
        return answer
    


class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        n=len(nums) #lignes
        m=len(nums[0]) #colonnes
        score=0
        for i in range(n):
            nums[i].sort(reverse=True)
        for j in range(m):
            maximum=0
            for i in range(n):
                maximum=max(nums[i][j],maximum)
            score+=maximum
        return score
    


class Solution:
    def transpose(self, grid: List[List[int]]) -> List[List[int]]:
        n=len(grid)
        m=len(grid[0])
        answer=[]
        for j in range(m):
            liste=[]
            for i in range(n):
                liste.append(grid[i][j])
            answer.append(liste)
        return answer 

    def equalPairs(self, grid: List[List[int]]) -> int:
        m=len(grid)
        n=len(grid[0])
        transposee=self.transpose(grid)
        number=0
        for i in range(m):
            for j in range(m):
                if transposee[i]==grid[j]:
                    number+=1
        return number
    


class Solution:
    def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
        m=len(grid)
        n=len(grid[0])
        diff=[[0 for _ in range(n)] for _ in range(m)]
        row=defaultdict(list)
        column=defaultdict(list)
        for i in range(m):
            number=grid[i].count(0)
            row[i].append(number)
            row[i].append(n-number)
        for j in range(n):
            number=sum([grid[i][j] for i in range(m)])
            column[j].append(m-number)
            column[j].append(number)
        for i in range(m):
            for j in range(n):
                diff[i][j]=row[i][1]+column[j][1]-row[i][0]-column[j][0]
        return diff
    


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n=len(matrix)
        copy_matrix=copy.deepcopy(matrix)
        for i in range(n):
            matrix[i]=list(reversed([copy_matrix[j][i] for j in range(n)]))




class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m=len(mat)
        n=len(mat[0])
        reshape=[]
        if m*n!=r*c:
            return mat
        liste=[mat[i][j] for i in range(m) for j in range(n)]
        for i in range(r):
            ligne=liste[i*c:(i+1)*c]
            reshape.append(ligne)
        return reshape
    


class Solution:
    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        if len(original)!=m*n:
            return []
        reshape=[]
        for i in range(m):
            reshape.append(original[i*n:(i+1)*n])
        return reshape
    

class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        m=len(mat)
        n=len(mat[0])
        number=0
        for row in mat:
            if sum(row)==1:
                indice=row.index(1)
                if sum([mat[j][indice] for j in range(m)])==1:
                    number+=1
        return number
    


class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        occurence=defaultdict(int)
        for i,liste in enumerate(mat):
            occurence[i]=sum(liste)
        liste=sorted(occurence.keys(), key=lambda x: occurence[x])
        return liste[0:k]
    


class Solution:
    def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
        m=len(grid)
        n=len(grid[0])
        answer=[0 for _ in range(n)]
        for i in range(n):
            maximum=0
            for j in range(m):
                maximum=max(maximum,len(str(grid[j][i])))
            answer[i]=maximum
        return answer
    



class Solution:
    def modifiedMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        answer=matrix 
        m=len(answer)
        n=len(answer[0])
        for i in range(m):
            for j in range(n):
                if answer[i][j]==-1:
                    answer[i][j]=max([answer[l][j] for l in range(m)])
        return answer  



class Solution:
    def critere(self, dictionnaire:dict, word:str) -> bool:
        temp_dict=dictionnaire.copy()
        for char in word:
            if char in temp_dict:
                temp_dict[char]-=1
        return all(value<=0 for value in temp_dict.values())

    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        words=sorted(words, key= lambda x:len(x))
        occurence=defaultdict(int)
        letters=[char.lower() for char in licensePlate if char.isalpha()]
        for letter in letters:
            occurence[letter]+=1
        for word in words:
            if self.critere(occurence,word):
                return word


#Version naive TLE
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if s2==s1:
            return True
        permutation=s1[::-1]
        if permutation in s2:
            return True
        m=len(s1)
        for i in range(len(s2)):
            if s2[i] in s1:
                if sorted(s1)==sorted(s2[i:i+m]):
                    return True
        return False
    

#1e version
class Solution:
    def findReplaceString(self, s: str, indices: List[int], sources: List[str], targets: List[str]) -> str:
        string=s
        for i,index in enumerate(indices):
            m=len(sources[i])
            if string[index:index+m]==sources[i]:
                s=s[:index]+targets[i]+s[index+m:]
        return s
    


class Solution:
    def findReplaceString(self, s: str, indices: List[int], sources: List[str], targets: List[str]) -> str:
        replacement=sorted(zip(indices,sources,targets), reverse= True)
        for indice, source, target in replacement:
            m=len(source)
            if s[indice:indice+m]==source:
                s=s[:indice]+target+s[indice+m:]
        return s
    


class Solution:
    def checkArithmetic(self, nums: List[int]) -> bool:
        nums.sort()
        diff=nums[1]-nums[0]
        for i in range(2,len(nums)):
            if nums[i]-nums[i-1]!=diff:
                return False
        return True

    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        m=len(l)
        answer=[False]*m
        for i in range(m):
            liste=nums[l[i]:r[i]+1]
            if self.checkArithmetic(liste):
                answer[i]=True
        return answer
    


# on peut etre plus elegant en ecrivant : liste.append(dictionnaire.get(word, "?"))
class Solution:
    def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
        dictionnaire={key:val for key,val in knowledge}
        liste=[]
        i=0
        while i<len(s):
            if s[i]=='(':
                j=i+1
                while s[j]!=')':
                    j+=1
                word=s[i+1:j]
                if word in dictionnaire:
                    liste.append(dictionnaire[word])
                else:
                    liste.append('?')
                i=j+1
            else:
                liste.append(s[i])
                i+=1
        return ''.join(liste)
    



class Solution:
    def isGood(self, s:str) -> bool:
        frequency=defaultdict(int)
        for char in s:
            frequency[char]+=1
            if frequency[char]>1:
                return False
        return True

    def countGoodSubstrings(self, s: str) -> int:
        n=len(s)
        count=0
        for i in range(n-2):
            if self.isGood(s[i:i+3]):
                count+=1
        return count
    



class Solution:
    def largestWordCount(self, messages: List[str], senders: List[str]) -> str:
        count=defaultdict(int)
        for i,name in enumerate(senders):
            count[name]+=len(messages[i].split())
        names=sorted(count.keys(), key=lambda x: (count[x],x), reverse=True)
        return names[0]
    



class Solution:
    def partitionString(self, s: str) -> int:
        count=0
        seen=set()
        for char in s:
            if char in seen:
                count+=1
                seen.clear()
            seen.add(char)
        return count+1
    



class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        n=len(s)
        vowels='aeiou'
        current_count=0
        for i in range(k):
            if s[i] in vowels:
                current_count+=1
        max_count=current_count
        for i in range(k,n):
            if s[i] in vowels:
                current_count+=1
            if s[i-k] in vowels:
                current_count-=1
            max_count=max(max_count,current_count)
        return max_count
    



class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        skill.sort()
        n=len(skill)
        ref_skill=skill[0]+skill[-1]
        chemistry=skill[0]*skill[-1]
        for i in range(1,n//2):
            ref=skill[i]+skill[n-1-i]
            if ref!=ref_skill:
                return -1
            chemistry+=skill[i]*skill[n-1-i]
        return chemistry



class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        candidats=[s[i:i+2] for i in range(len(s)-1)]
        for candidat in candidats:
            if candidat in s[::-1]:
                return True
        return False
    


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n=len(s)
        for i in range(n//2):
            candidat=s[:i+1]
            m=len(candidat)
            puissance=n//m
            if candidat*puissance==s:
                return True
        return False



class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        prime=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        index=[]
        for i,num in enumerate(nums):
            if num in prime:
                index.append(i)
        return abs(index[0]-index[-1])




class Solution:
    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        n=len(s)
        i=0
        answer=[]
        while i<n:
            word=s[i:i+k]
            if len(word)==k:
                answer.append(word)
            else:
                word=word+(k-len(word))*fill
                answer.append(word)
            i+=k
        return answer




class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        n=len(candies)
        maximum=max(candies)
        result=[False]*n
        for i,candy in enumerate(candies):
            if candy+extraCandies>=maximum:
                result[i]=True
            else:
                result[i]=False
        return result



class Solution:
    def match(self, s: string, pattern: str) -> bool:
        n=len(pattern)
        indexes=[]
        words=[]
        if [char for char in s if char.isupper()]!=[char for char in pattern if char.isupper()]:
            return False
        for i,char in enumerate(pattern):
            if char.isupper():
                indexes.append(i)
        for i in range(len(indexes)-1):
            words.append(pattern[indexes[i]:indexes[i+1]])
        if pattern[-1].isupper():
            words.append(pattern[-1])
        for word in words:
            if word not in s:
                return False
        return True
        

    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
        answer=[False]*len(queries)
        for i in range(len(queries)):
            if self.match(queries[i],pattern):
                answer[i]=True
        return answer



class Solution:
    def destCity(self, paths: List[List[str]]) -> str:
        cities=[paths[i][1] for i in range(len(paths))]
        departures=[paths[i][0] for i in range(len(paths))]
        for city in cities:
            if city not in departures:
                return city


class Solution:
    def greatestLetter(self, s: str) -> str:
        majuscule=defaultdict(int)
        minuscule=defaultdict(int)
        for char in s:
            if char.isupper():
                majuscule[char]+=1
            else:
                minuscule[char]+=1
        candidats=[key for key in minuscule.keys() if key.upper() in majuscule.keys()]
        candidats=sorted(candidats,reverse=True)
        if candidats:
            return candidats[0].upper()
        else:
            return ""



class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        count=0
        for arr in arr1:
            drapeau=True
            for element in arr2:
                if abs(arr-element)<=d:
                    drapeau=False
                    break
            if drapeau:
                count+=1
        return count



class Solution:
    def stableMountains(self, height: List[int], threshold: int) -> List[int]:
        indexes=[]
        for i in range(1,len(height)):
            if height[i-1]>threshold:
                indexes.append(i)
        return indexes 




class Solution:
    def encrypt(self, n:int) -> int:
        x=str(n)
        m=max(x)
        x=m*len(x)
        return int(x)

    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        return sum([self.encrypt(num) for num in nums])



#Tres content car medium a 66% de reussite et fait en 2min
class Solution:
    def distance(self, liste: List[int]) -> int:
        x,y=liste
        return x**2+y**2

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points=sorted(points, key=lambda x:self.distance(x))
        return points[0:k]




class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        apples=sum(apple)
        capacity=sorted(capacity, reverse=True)
        boxes=0
        i=0
        while apples>0:
            apples-=capacity[i]
            boxes+=1
            i+=1
        return boxes


class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        altitudes=[0]
        past=0
        for denivele in gain:
            altitudes.append(past+denivele)
            past+=denivele
        return max(altitudes)




class Solution:
    def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
        n=len(nums)
        distant=[]
        indexes=[j for j in range(n) if nums[j]==key]
        for i in range(n):
            if any(abs(i-j)<=k for j in indexes):
                distant.append(i)
        return distant


#Degueulasse
class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        good=0
        n=len(arr)
        for i in range(n-2):
            for j in range(i+1,n-1):
                for k in range(j+1,n):
                    if abs(arr[i]-arr[j])<=a and abs(arr[j]-arr[k])<=b and abs(arr[i]-arr[k])<=c:
                        good+=1
        return good




class Solution:
    def getMinDistance(self, nums: List[int], target: int, start: int) -> int:
        n=len(nums)
        indexes=[i for i in range(n) if nums[i]==target]
        minimum=float('inf')
        for j in indexes:
            minimum=min(minimum,abs(j-start))
        return minimum




class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        index=-1
        manhattan=float('inf')
        for i,point in enumerate(points):
            x1,y1=point
            if x1==x or y1==y:
                if abs(x-x1)+abs(y-y1)<manhattan:
                    manhattan=abs(x-x1)+abs(y-y1)
                    index=i
        return index


class Solution:
    def operation(self, num:int) -> int:
            x=str(num)
            return sum([int(char) for char in x])

    def minElement(self, nums: List[int]) -> int:
        return min([self.operation(num) for num in nums])



class Solution:
    def splitNum(self, num: int) -> int:
        num1=""
        num2=""
        x=sorted(str(num))
        for i,char in enumerate(x):
            if i%2==0:
                num1+=char
            else:
                num2+=char
        return int(num1) + int(num2)




#TRIVIAL
class Solution:
    def convertTemperature(self, celsius: float) -> List[float]:
        kelvin=celsius+273.15
        fahr=celsius*1.80 + 32.00
        return [kelvin,fahr]


class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        j=0
        while j<k:
            x=min(nums)
            min_index=nums.index(x)
            nums[min_index]=x*multiplier
            j+=1
        return nums



class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        columns=0
        n=len(strs[0])
        for i in range(n):
            for j in range(1,len(strs)):
                if strs[j-1][i]>strs[j][i]:
                    columns+=1
                    break
        return columns


class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        words=list(s.split())[0:k]
        return ' '.join(words)



class Solution:
    def distanceBetweenBusStops(self, distance: List[int], start: int, destination: int) -> int:
        if start>destination:
            start,destination=destination,start
        distance1=sum(distance[start:destination])
        distance2=sum(distance[destination:])+sum(distance[:start])
        return min(distance1,distance2)



class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        occurence=defaultdict(int)
        for num in arr:
            occurence[num]+=1
        sorted_occurence=sorted(occurence.items(),key=lambda x:x[1])
        reste=len(set(arr))
        for key,val in sorted_occurence:
            if k>=val:
                k-=val
                reste-=1
            else:
                break
        return reste



class Solution:
    def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
        matrix=[[0]*n for _ in range(m)]
        for row,column in indices:
            for j in range(n):
                matrix[row][j]+=1
            for i in range(m):
                matrix[i][column]+=1
        odd=0
        for row in matrix:
            for num in row:
                if num%2!=0:
                    odd+=1
        return odd


class Solution:
    def minimumDeletions(self, nums: List[int]) -> int:
        n=len(nums)
        mediane=n//2
        max_index=nums.index(max(nums))
        min_index=nums.index(min(nums))
        if max_index<=mediane and min_index<=mediane:
            return max(max_index,min_index)+1
        elif max_index>=mediane and min_index>=mediane:
            return n-min(max_index,min_index)
        else:
            front=max(min_index,max_index)+1
            back=n-min(min_index,max_index)
            mix=min(min_index+1+n-max_index, n-min_index+max_index+1)
            return min(front,back,mix)



class Solution:
    def commonFactors(self, a: int, b: int) -> int:
        common=0
        for i in range(1,min(a,b)+1):
            if a%i==0 and b%i==0:
                common+=1
        return common



class Solution:
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        s=list(s)
        final=[]
        spaces_index=0
        for i,char in enumerate(s):
            if spaces_index<len(spaces) and i==spaces[spaces_index]:
                final.append(' ')
                final.append(char)
                spaces_index+=1
            else:
                final.append(char)
        return ''.join(final)



Version 1 du precedent: TLE 59/66
class Solution: 
	def addSpaces(self, s: str, spaces: List[int]) -> str: 
		s=list(s) 
		final=[] 
		for i,char in enumerate(s): 
			if i in spaces: 	
				final.append(' ') 
				final.append(char) 
			else: 
				final.append(char) 
		return ''.join(final)


#Pas terrible du tout pour la complexité, bat seulement 13% des soumissions
class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        occurence=defaultdict(int)
        for element in arr:
            occurence[element]+=1
        for element in target:
            occurence[element]-=1
        if list(set(occurence.values()))!=[0]:
            return False
        return True



class Solution:
    def maximumBags(self, capacity: List[int], rocks: List[int], additionalRocks: int) -> int:
        n=len(capacity)
        remaining_place=sorted([capacity[i]-rocks[i] for i in range(n)])
        full_bags=0
        i=0
        while additionalRocks>0 and i<n:
            if remaining_place[i]>0:
                fill_amount=min(additionalRocks,remaining_place[i])
                remaining_place[i]-=fill_amount
                additionalRocks-=fill_amount
            i+=1
        return remaining_place.count(0)



class Solution:
    def countAsterisks(self, s: str) -> int:
        asterisks=0
        bars=0
        for char in s:
            if char=='|':
                bars+=1
            if char=='*' and bars%2==0:
                asterisks+=1
        return asterisks


#Runtime 5% seulement : horrible
class Solution:
    def position(self, s:str) -> int:
        return ord(s)-ord('a')

    def getLetter(self, n:int) -> str:
        return chr(n+ord('a'))

    def stringHash(self, s: str, k: int) -> str:
        n=len(s)
        result=[]
        i=0
        while i<len(s):
            substring=s[i:i+k]
            hashedChar=sum([self.position(substring[i]) for i in range(len(substring))])%26
            identification=self.getLetter(hashedChar)
            result.append(identification)
            i+=k
        return ''.join(result)



class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        even=sum(num for num in nums if num % 2 == 0)
        answer=[]
        for val,index in queries:
            if nums[index]%2==0:
                even-=nums[index] 
            nums[index]+=val
            if nums[index]%2==0:
                even+=nums[index]  
            answer.append(even)
        return answer



#Runtime 5% : pas terrible du tout. Autre approche apres
class Solution:
    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
        matching=0
        players=sorted(players)
        trainers=sorted(trainers)
        for player in players:
            for trainer in trainers:
                if trainer>=player:
                    matching+=1
                    trainers.remove(trainer)
                    break
        return matching



#40% runtime : two pointers bcp mieux
class Solution:
    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
        matching=0
        players=sorted(players)
        trainers=sorted(trainers)
        i,j=0,0
        while i<len(players) and j<len(trainers):
            if players[i]<=trainers[j]:
                matching+=1
                i+=1
            j+=1
        return matching



#Enonce mal compris : fait qu’avec les puissances de 2
class Solution:
    def powerOfTwo(self,n:int)->bool:
        if n%2!=0:
            return False
        while n%2==0 and n>1:
            n//=2
        return n==1

    def longestSquareStreak(self, nums: List[int]) -> int:
        nums=sorted(nums)
        powers=sorted([num for num in nums if self.powerOfTwo(num)])
        if len(powers)<2:
            return -1
        streak=1
        for power in powers:
            if power*power in powers:
                streak+=1
        if streak<2:
            return -1
        return streak


#J’ABANDONNE
class Solution:
    def longestSquareStreak(self, nums: List[int]) -> int:
        longest=0
        for num in set(nums):
            streak=1
            current=num
            while current*current in set(nums):
                streak+=1
                current=current*current
            longest=max(longest,streak)
            current = num
            while current in nums:
                nums.remove(current)
                current = current * current
        if longest<2:
            return -1
        return longest


#Celui là marche
class Solution:
    def longestSquareStreak(self, nums: List[int]) -> int:
        longest=0
        nums=set(sorted(nums))
        for num in nums:
            streak=1
            current=num
            while current*current in nums:
                streak+=1
                current=current*current
            longest=max(longest,streak)
        if longest<2:
            return -1
        return longest



class Solution:
    def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
        n=len(logs)
        times=[logs[0][1]]
        for i in range(1,n):
            times.append(logs[i][1]-logs[i-1][1])
        maximum_time=max(times)
        candidates=[]
        for i,time in enumerate(times):
            if time==maximum_time:
                candidates.append(logs[i][0])
        return min(candidates)


class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        m=len(mat)
        n=len(mat[0])
        number=0
        for row in mat:
            if sum(row)==1:
                indice=row.index(1)
                if sum([mat[j][indice] for j in range(m)])==1:
                    number+=1
        return number



class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        longest=0
        i=0
        while i<len(s):
            length=0
            seen=set()
            j=i
            while j<len(s) and s[j] not in seen:
                seen.add(s[j])
                j+=1
                length+=1
            longest=max(longest,length)
            i+=1
        return longest 



#ONE LINER : BOOOM. 36,3% d’acceptation
class Solution:
    def countSegments(self, s: str) -> int:
        return len(s.split())


class Solution:
    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        n=len(s)
        total_pixels=0
        lines=1
        for i in range(n):
            index=ord(s[i])-ord('a')
            if total_pixels+widths[index]>100:
                lines+=1
                total_pixels=widths[index]
            else:
                total_pixels+=widths[index]
        return [lines, total_pixels]


class Solution:
    def compressedString(self, word: str) -> str:
        comp=[]
        word=list(word)
        n=len(word)
        i=0
        while i<n:
            j=i
            current_count=1
            while j<len(word)-1 and word[j+1]==word[j]:
                current_count+=1
                j+=1
            if current_count<=9:
                comp.append(str(current_count))
                comp.append(word[i])
            else:
                q=current_count//9
                r=current_count%9
                comp.extend(['9'+word[i]]*q)
                if r>0:
                    comp.append(str(r))
                    comp.append(word[i])
            i=j+1
        return ''.join(comp)



class Solution:
    def closetTarget(self, words: List[str], target: str, startIndex: int) -> int:
        if target not in words:
            return -1
        shortest=float('inf')
        n=len(words)
        liste_index=[i for i in range(len(words)) if words[i]==target]
        for index in liste_index:
            left_to_right=(index-startIndex)%n
            right_to_left=(startIndex-index)%n
            distance=min(left_to_right,right_to_left)
            shortest=min(shortest, distance)
        return shortest



class Solution:
    def squareIsWhite(self, coordinates: str) -> bool:
        letter=ord(coordinates[0])-ord('a')+1
        figure=int(coordinates[1])
        if (letter+figure)%2!=0:
            return True
        return False



lass Solution:
    def sumDigit(self, n:int) ->int:
        somme=0
        while n>0:
            somme+=n%10
            n//=10
        return somme

    def maximumSum(self, nums: List[int]) -> int:
        n=len(nums)
        somme=defaultdict(list)
        for num in nums:
            somme[self.sumDigit(num)].append(num)
        if all(len(val)<2 for val in somme.values()):
            return -1
        maximum=float('-inf')
        candidates=[val for val in somme.values() if len(val)>=2]
        for val in candidates:
            val=sorted(val, reverse=True)
            maximum=max(maximum, sum(val[:2]))
        return maximum



class Solution:
    def sequence(self, s:str, t:str):
        first_position=ord(s)-ord('a')
        second_position=ord(t)-ord('a')
        return second_position-first_position==1

    def longestContinuousSubstring(self, s: str) -> int:
        i=0
        longest=0
        while i<len(s):
            j=i
            length=1
            while j<len(s)-1 and self.sequence(s[j],s[j+1]):
                j+=1
                length+=1
            longest=max(longest,length)
            i=j+1
        return longest



class Solution:        
    def printVertically(self, s: str) -> List[str]:
        s=s.split()
        m=max(len(word) for word in s)
        answer=['']*m
        for i in range(m):
            for word in s:
                if i>=len(word):
                    answer[i]+=' '
                else:
                    answer[i]+=word[i]
        return [word.rstrip() for word in answer]


#Convertir les lists en sets aident contre le TLE : acces en 0(1)
class Solution:
    def topStudents(self, positive_feedback: List[str], negative_feedback: List[str], report: List[str], student_id: List[int], k: int) -> List[int]:
        positive_feedback=set(positive_feedback)
        negative_feedback=set(negative_feedback)
        score=defaultdict(int)
        for i,student in enumerate(student_id):
            rapport=report[i].split()
            count=0
            for word in rapport:
                if word in positive_feedback:
                    count+=3
                elif word in negative_feedback:
                    count-=1
            score[student]=count
        students=sorted(list((score.keys())), key=lambda x:(-score[x],x))
        return students[:k]


class Solution:        
    def printVertically(self, s: str) -> List[str]:
        s=s.split()
        m=max(len(word) for word in s)
        answer=['']*m
        for i in range(m):
            for word in s:
                if i>=len(word):
                    answer[i]+=' '
                else:
                    answer[i]+=word[i]
        return [word.rstrip() for word in answer]



#Convertir les lists en sets aident contre le TLE : acces en 0(1)
class Solution:
    def topStudents(self, positive_feedback: List[str], negative_feedback: List[str], report: List[str], student_id: List[int], k: int) -> List[int]:
        positive_feedback=set(positive_feedback)
        negative_feedback=set(negative_feedback)
        score=defaultdict(int)
        for i,student in enumerate(student_id):
            rapport=report[i].split()
            count=0
            for word in rapport:
                if word in positive_feedback:
                    count+=3
                elif word in negative_feedback:
                    count-=1
            score[student]=count
        students=sorted(list((score.keys())), key=lambda x:(-score[x],x))
        return students[:k]


class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        units=0
        boxTypes=sorted([liste for liste in boxTypes], key= lambda x:-x[1])
        n=len(boxTypes)
        i=0
        while truckSize>0 and i<n:
            box,unit=boxTypes[i]
            fill_boxes=min(box,truckSize)
            truckSize-=fill_boxes
            units+=fill_boxes*unit
            i+=1
        return units



class Solution:
    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        n=len(arr)
        m=sorted(arr)[(n-1)//2]
        arr=sorted(arr, key=lambda x:(-abs(x-m),-x))
        return arr[:k]

#deja faits mais mieux réalisés
class Solution:
    def frequencySort(self, s: str) -> str:
        word=''
        frequency=defaultdict(int)
        for char in s:
            frequency[char]+=1
        couples=sorted(frequency.items(), key=lambda x:-x[1])
        for char,iteration in couples:
            word+=char*iteration
        return word



class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if word.upper()==word or word.lower()==word:
            return True
        else:
            if not word[0].isupper():
                return False
            for i in range(1,len(word)):
                if word[i].isupper():
                    return False
        return True


class Solution:
    def countSeniors(self, details: List[str]) -> int:
        old=0
        for passenger in details:
            age=int(passenger[11:13])
            if age>60:
                old+=1
        return old


class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        enfants=sorted(g)
        cookies=sorted(s)
        content=0
        i,j=0,0
        while i<len(enfants) and j<len(cookies):
            if enfants[i]<=cookies[j]:
                content+=1
                i+=1
            j+=1
        return content

class Solution:
    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        keyboard=0
        text=text.split()
        for word in text:
            if all(char not in brokenLetters for char in word):
                keyboard+=1
        return keyboard


class Solution:
    def minTimeToType(self, word: str) -> int:
        n=len(word)
        number=min((ord(word[0])-ord('a'))%26,(ord('a')-ord(word[0]))%26)
        for i in range(1,n):
            indice1=ord(word[i-1])-ord('a')
            indice2=ord(word[i])-ord('a')
            number+=min((indice2-indice1)%26,(indice1-indice2)%26)
        return number+n


class Solution:
    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        occurence=defaultdict(int)
        for num in nums:
            occurence[num]+=1
        keys=[key for key in occurence.keys() if occurence[key]==2]
        return keys


#Meilleur runtime
class Solution:
    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        keys=[]
        occurence=defaultdict(int)
        for num in nums:
            occurence[num]+=1
            if occurence[num]==2:
                keys.append(num)
        return keys



#100% en terme de complexite temps 
class Solution:
    def countKeyChanges(self, s: str) -> int:
        count=0
        for i in range(1,len(s)):
            if s[i] not in [s[i-1].lower(), s[i-1].upper()]:
                count+=1
        return count


class Solution:
    def isBalanced(self, num: str) -> bool:
        even=sum([int(num[i]) for i in range(len(num)) if i%2==0])
        odd=sum([int(num[i]) for i in range(len(num)) if i%2!=0])
        return even==odd



class Solution:
    def interpret(self, command: str) -> str:
        answer=[]
        i=0
        while i<len(command):
            if command[i]=='G':
                answer.append('G')
                i+=1
            elif command[i]=='(' and command[i+1]==')':
                answer.append('o')
                i+=2
            else:
                answer.append('al')
                i+=4
        return ''.join(answer)



  def reformatDate(self, date: str) -> str:
        answer=''
        date=date.split()
        months = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06", "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
        day,month,year=date[0],date[1],date[2]
        answer+=year + '-'
        month=months[month]
        answer+=month + '-'
        if len(day)==3:
            day=day[:1]
            answer+='0' + day 
        else:
            day=day[:2]
            answer+=day
        return answer



class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count=0
        for i in range(len(nums)-1):
            if nums[i]>=nums[i+1]:
                ecart=nums[i]-nums[i+1]
                count+=ecart+1
                nums[i+1]+=ecart+1
        return count



class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages=[]
        for _ in range(len(nums)//2):
            minimum=min(nums)
            nums.remove(minimum)
            maximum=max(nums)
            nums.remove(maximum)
            averages.append((minimum+maximum)/2)
        return min(averages)


class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count=0
        for integer in range(low,high+1):
            x=str(integer)
            if len(x)%2==0:
                n=len(x)//2
                first_half=[int(x[i]) for i in range(0,n)]
                second_half=[int(x[i]) for i in range(n,len(x))]
                if sum(first_half)==sum(second_half):
                    count+=1
        return count



class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        occurence = defaultdict(int)
        for num in nums:
            occurence[num]+=1
        for key, val in occurence.items():
            if val==1:
                return key

class Solution:
    def frequencySort(self, s: str) -> str:
        word =''
        frequency = defaultdict(int)
        for char in s:
            frequency[char] += 1
        keys = sorted(frequency.keys(), key= lambda x: -frequency[x])
        for key in keys:
            word += key*frequency[key]
        return word



class Solution:
    def frequencySort(self, s: str) -> str:
        word =''
        frequency = defaultdict(int)
        for char in s:
            frequency[char] += 1
        keys = sorted(frequency.keys(), key= lambda x: -frequency[x])
        for key in keys:
            word += key*frequency[key]
        return word


class Solution:
    def fib(self, n: int) -> int:
        if n==0:
            return 0
        if n==1:
            return 1
        answer = [0, 1]
        for i in range(2,n+1):
            answer.append(answer[i-1] + answer[i-2])
        return answer[n]



class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {1:1, 2:2}
        def f(n):
            if n in memo:
                return memo[n]
            else:
                memo[n] = f(n-1) + f(n-2)
            return memo[n]
        return f(n)


def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        memo = {0:0, 1:0} # best cases

        def min_cost(i):
            if i in memo:
                return memo[i]
            else:
                memo[i] = min(cost[i-2] + min_cost(i-2), cost[i-1] + min_cost(i-1))
                return memo[i]
        
        return min_cost(n)


# Version 2 : mieux 

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp = [0]*(n+1)
        for i in range(2, n+1):
            dp[i] = min(cost[i-1] + dp[i-1], cost[i-2] + dp[i-2])
        return dp[n]






