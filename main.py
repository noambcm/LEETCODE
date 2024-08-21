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
    



    





    











 

