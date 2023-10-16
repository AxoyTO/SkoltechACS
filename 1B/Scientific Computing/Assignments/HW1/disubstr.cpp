#include <algorithm>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr int MAXL = 1024;  // Max length of the string
constexpr int MAXP = 12;    // Max power of 2

// Arrays to store the string at each level and the inverse of the suffix array
int stringLevels[MAXP][MAXL], inverseSuffixArray[MAXL];

// Node of the suffix array, storing index and two ranks
class SuffixNode {
 public:
  int index;
  int ranks[2];

  void display() {
    std::cout << "Index: " << index << std::endl;
    std::cout << "Ranks: ";
    for (auto& i : ranks) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
  }
};

// Suffix array
SuffixNode suffixArray[MAXL];

// Number of steps or powers of 2 and length of the string
int currentStep, stringLength;

// Compare function to compare two SuffixNodes
bool compareSuffixNodes(const SuffixNode& a, const SuffixNode& b) {
  return a.ranks[0] == b.ranks[0] ? (a.ranks[1] < b.ranks[1])
                                  : (a.ranks[0] < b.ranks[0]);
}

/* The function starts by initializing the first level of stringLevels with the
 * ASCII values of the characters in the input string. It then enters a loop
 * that constructs the suffix array using the prefix doubling algorithm. This
 * algorithm involves sorting the suffixes based on their ranks at each step.
 * For each suffix, it calculates two ranks: one based on the characters at the
 * current position and another based on the characters after moving to the next
 * level (doubling the step size). After sorting the suffixes, it updates the
 * ranks for the current level and stores these ranks in stringLevels. The loop
 * continues until reaching the power of 2 larger than the string length.*/
void SA(const std::string& inputString) {
  // Initialize the first level with the ASCII values of the characters in the
  // string
  for (size_t i = 0; i < inputString.length(); i++)
    stringLevels[0][i] = inputString[i];

  // Loop until we reach the power of 2 larger than the string length
  currentStep = 1;
  stringLength = inputString.length();

  for (int step = 1; (1 << (step - 1)) < stringLength; step++, currentStep++) {
    // Loop over all suffixes
    for (int j = 0; j < stringLength; j++) {
      // First half will be the first rank the one which we have already sorted
      suffixArray[j].ranks[0] = stringLevels[currentStep - 1][j];

      // Next half will be second rank and other suffix which we have already
      // sorted in previous step for this k = 4, 2 and 2 length strings are
      // already sorted
      int nextHalfIndex = j + (1 << (step - 1));
      suffixArray[j].ranks[1] =
          (nextHalfIndex < stringLength)
              ? stringLevels[currentStep - 1][nextHalfIndex]
              : -1;

      // Store the current index as it will change after sorting
      suffixArray[j].index = j;
    }

    // Sort the suffix array based on ranks
    std::sort(suffixArray, suffixArray + stringLength, compareSuffixNodes);

    // Add new ranks of this level to each suffix
    for (int j = 0; j < stringLength; j++) {
      int prevIndex = suffixArray[j].index;
      int prevRank1 = suffixArray[j].ranks[0];
      int prevRank2 = suffixArray[j].ranks[1];

      int prevRankIndex = (j > 0) ? suffixArray[j - 1].index : -1;
      int prevRank1Prev = (j > 0) ? suffixArray[j - 1].ranks[0] : -1;
      int prevRank2Prev = (j > 0) ? suffixArray[j - 1].ranks[1] : -1;

      // If the current suffix has same rank as previous one then do not
      // increase its rank Else increase it or make it equal to current j.
      stringLevels[currentStep][prevIndex] =
          (step > 0 &&
           (prevRank1 == prevRank1Prev && prevRank2 == prevRank2Prev))
              ? stringLevels[currentStep][prevRankIndex]
              : j;
    }
  }
}

/* The LCP function calculates the longest common prefix between two suffixes by
 * comparing their ranks in the stringLevels array. It iteratively checks if the
 * prefixes at each level (from highest to lowest power of 2) are equal and
 * increases the length of the common prefix accordingly. */
int LCP(int x, int y) {
  // if both are less than x return
  if (x < 0 || y < 0)
    return 0;

  // Loop until the power becomes 0 or numbers become greater than length
  // As extreme cases in search of binary search
  int result = 0;
  for (int k = currentStep - 1; k >= 0 && x < stringLength && y < stringLength;
       k--) {
    // If the prefixes at current power are equal
    // Increase the x and y values after the equal
    // prefixes. if they are at 0 and 0 and k is 4
    // increase than to 4 and 4 as first 4 characters are equal
    // According to level,now find the next 2 one if they are
    // equal or not.
    if (stringLevels[k][x] == stringLevels[k][y]) {
      x += (1 << k);
      y += (1 << k);
      result += (1 << k);
    }
  }

  return result;
}

/* The main function first reads the number of test cases.
 * For each test case, it reads an input string and initializes the data
 * structures. It then constructs the suffix array and computes the inverse of
 * the suffix array (i.e., a mapping from index to rank). After constructing the
 * suffix array and inverse, it calculates the total number of distinct
 * substrings. This is done by iterating through the suffixes in the sorted
 * order and subtracting the LCP of consecutive suffixes from the total string
 * length.
 * */
int main() {
  int numTestCases;
  std::cin >> numTestCases;
  std::vector<int> results;

  std::string inputString;

  double time_taken_sa = 0;
  double time_taken_invd = 0;

  auto start_total = clock();

  while (numTestCases--) {
    std::cin >> inputString;

    // Initialize all variables
    memset(stringLevels, 0, sizeof(stringLevels));
    memset(suffixArray, 0, sizeof(suffixArray));
    memset(inverseSuffixArray, 0, sizeof(inverseSuffixArray));

    currentStep = 0;

    auto start = clock();
    // Find the suffix array
    SA(inputString);
    auto stop = clock() - start;
    time_taken_sa += ((double)stop / CLOCKS_PER_SEC);

    start = clock();
    // Find the inverse of suffix array
    for (int i = 0; i < stringLength; i++)
      inverseSuffixArray[stringLevels[currentStep - 1][i]] = i;

    // Find the lcp of consecutive suffixes as it will give how many characters
    // can make distinct substring.
    int totalDistinctSubstrings = stringLength - suffixArray[0].index;
    for (int i = 1; i < stringLength; i++) {
      // Subtract from total length of current suffix the lcp of this and
      // previous one
      totalDistinctSubstrings +=
          (stringLength - suffixArray[i].index) -
          LCP(inverseSuffixArray[i - 1], inverseSuffixArray[i]);
    }
    stop = clock() - start;
    time_taken_invd += ((double)stop / CLOCKS_PER_SEC);

    results.emplace_back(totalDistinctSubstrings);
  }

  for (auto& result : results) {
    std::cout << result << std::endl;
  }

  auto stop_total = clock() - start_total;
  double total_time = ((double)stop_total) / CLOCKS_PER_SEC;

  std::cout << std::fixed;

  std::cout << "Time to construct SA: " << time_taken_sa << " s" << std::endl;
  std::cout << "Time to find inverse SA and total distinct substrings: "
            << time_taken_invd << " s" << std::endl;
  std::cout << "Total execution time: " << total_time << " s" << std::endl;

  return 0;
}
