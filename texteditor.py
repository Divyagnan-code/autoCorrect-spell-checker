import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinter.font import Font
import threading
from typing import Set, Dict
import re
import json
from datetime import datetime
import time
from ttkthemes import ThemedTk
from difflib import get_close_matches
import nltk
from nltk.corpus import words
from nltk.corpus import brown
from collections import Counter
import re
from typing import List, Dict, Set
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns


class Autocorrect:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('corpora/words')
            nltk.data.find('corpora/brown')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('words')
            nltk.download('brown')

        # Create vocabulary from both words and brown corpus
        word_list = words.words()
        brown_words = brown.words()

        self.words = set(word.lower() for word in word_list + list(brown_words) if word.isalpha())
        self.word_counts = Counter(word.lower() for word in brown_words if word.isalpha())

        # Initialize the model and vectorizer
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
        self.model = MultinomialNB()
        self.correction_stats = Counter()
        self.train_model()

    def train_model(self):
        """Train the machine learning model on sample misspelled and correct words."""
        # Example training data
        training_data = [
            ('speling', 'spelling'),
            ('recieve', 'receive'),
            ('definately', 'definitely'),
            ('occured', 'occurred'),
            ('seperate', 'separate'),
            # Add more pairs as needed
        ]

        # Prepare the data
        misspelled_words, correct_words = zip(*training_data)
        X = self.vectorizer.fit_transform(misspelled_words)
        self.model.fit(X, correct_words)

        # Update correction statistics
        for misspelled, correct in training_data:
            self.correction_stats[correct] += 1

        # Plot correction statistics
        self.plot_correction_statistics()

    def plot_correction_statistics(self):
        """Visualize the correction statistics."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(self.correction_stats.keys()), y=list(self.correction_stats.values()), palette="viridis")
        plt.title('Frequency of Corrections Made by the Model')
        plt.xlabel('Correct Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def predict_correction(self, word: str) -> str:
        """Predict the correction for a misspelled word using ML model."""
        word_vector = self.vectorizer.transform([word])
        prediction = self.model.predict(word_vector)
        return prediction[0]

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _get_candidates(self, word: str) -> Set[str]:
        """Generate possible spelling corrections for a word."""
        candidates = {word}  # Start with the original word
        if word in self.words:
            return candidates  # If correct, return it

        # Get candidates with edit distance 1
        candidates.update(self._get_candidates_with_edit_distance(word))

        # Use ML model to predict correction
        if word not in candidates:
            ml_correction = self.predict_correction(word)
            candidates.add(ml_correction)
            self.correction_stats[ml_correction] += 1  # Update correction stats

        return {word for word in candidates if word in self.words}

    def _get_candidates_with_edit_distance(self, word: str) -> Set[str]:
        """Get candidates using edit distance method."""
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        # Generate all possible edits
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        # Combine all edits and filter for known words
        return set(deletes + transposes + replaces + inserts)

    def correct_word(self, word: str) -> str:
        """Correct a single word using ML and dictionary lookup."""
        word = word.lower()
        if word in self.words:
            return word
        candidates = self._get_candidates(word)
        if not candidates:
            return word
        # Sort candidates by frequency in Brown corpus
        return max(candidates, key=lambda x: self.word_counts.get(x, 0))

    def correct_text(self, text: str) -> str:
        """Correct all words in a text."""
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        corrected_words = []
        for word in words:
            if word.isalpha():
                corrected_words.append(self.correct_word(word))
            else:
                corrected_words.append(word)
        return ' '.join(corrected_words)


class SpellCheckEditor:
    def __init__(self):
        # Initialize main window with themed widgets
        self.root = ThemedTk(theme="arc")  # Modern theme
        self.root.title("Advanced Spell-Check Editor")
        self.root.geometry("1000x800")

        # Initialize status variables first
        self.status_var = tk.StringVar(value="Ready")
        self.word_count_var = tk.StringVar(value="Words: 0")

        # Initialize the autocorrect system in a separate thread
        self.autocorrect = None
        self.init_autocorrect()

        # Configuration
        self.check_interval = 1000  # ms
        self.last_check_time = 0
        self.misspelled_words: Dict[str, Set[int]] = {}

        self.setup_styles()
        self.create_gui()
        self.setup_bindings()

        # Start periodic spell checking
        self.check_spelling()

    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        style.configure("Modern.TButton",
                        padding=10,
                        font=('Helvetica', 10))
        style.configure("Status.TLabel",
                        padding=5,
                        background='#f0f0f0')

    def create_gui(self):
        """Create the graphical user interface"""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create toolbar
        self.create_toolbar()

        # Create main content area with text editor
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Create and configure text editor
        self.create_editor()

        # Create status bar
        self.create_status_bar()

        # Create right-click menu
        self.create_context_menu()

    def create_toolbar(self):
        """Create the toolbar with buttons"""
        toolbar = ttk.Frame(self.main_container)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        # File operations
        ttk.Button(toolbar, text="New", style="Modern.TButton",
                   command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open", style="Modern.TButton",
                   command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", style="Modern.TButton",
                   command=self.save_file).pack(side=tk.LEFT, padx=2)

        # Editing operations
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(toolbar, text="Check Spelling", style="Modern.TButton",
                   command=self.force_spell_check).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Auto-Correct All", style="Modern.TButton",
                   command=self.autocorrect_all).pack(side=tk.LEFT, padx=2)

    def create_editor(self):
        """Create the main text editor area"""
        self.text_widget = scrolledtext.ScrolledText(
            self.content_frame,
            wrap=tk.WORD,
            font=('Consolas', 12),
            background='#ffffff',
            foreground='#000000',
            insertbackground='#000000',
            selectbackground='#a6a6a6',
            selectforeground='#ffffff',
            width=80,
            height=30
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        # Configure tags for misspelled words
        self.text_widget.tag_configure("misspelled", background="#ffe6e6")

    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self.main_container)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(status_frame, textvariable=self.status_var,
                  style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.word_count_var,
                  style="Status.TLabel").pack(side=tk.RIGHT)

    def create_context_menu(self):
        """Create the right-click context menu"""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.suggestion_menu = tk.Menu(self.context_menu, tearoff=0)
        self.context_menu.add_cascade(label="Suggestions", menu=self.suggestion_menu)

    def init_autocorrect(self):
        """Initialize the autocorrect system in a separate thread"""

        def init_thread():
            self.status_var.set("Initializing spell checker...")
            self.autocorrect = Autocorrect()
            self.status_var.set("Ready")

        thread = threading.Thread(target=init_thread)
        thread.start()

    def check_spelling(self):
        """Perform spell checking on the text"""
        if self.autocorrect is None:
            self.root.after(1000, self.check_spelling)
            return

        current_time = time.time()
        if current_time - self.last_check_time < 1.0:  # Throttle checking
            self.root.after(self.check_interval, self.check_spelling)
            return

        self.last_check_time = current_time

        # Clear previous misspellings
        for tag in self.text_widget.tag_names():
            if tag == "misspelled":
                self.text_widget.tag_remove(tag, "1.0", tk.END)

        # Get text and check words
        text = self.text_widget.get("1.0", tk.END)
        words = re.findall(r'\b\w+\b', text)

        self.misspelled_words.clear()

        for word in words:
            if word.isalpha() and word.lower() not in self.autocorrect.words:
                # Find all occurrences of the word
                start_idx = "1.0"
                while True:
                    start_idx = self.text_widget.search(
                        word, start_idx, tk.END, nocase=False)
                    if not start_idx:
                        break
                    end_idx = f"{start_idx}+{len(word)}c"
                    self.text_widget.tag_add("misspelled", start_idx, end_idx)
                    start_idx = end_idx

        # Update word count
        self.update_word_count()

        # Schedule next check
        self.root.after(self.check_interval, self.check_spelling)

    def update_word_count(self):
        """Update the word count in the status bar"""
        text = self.text_widget.get("1.0", tk.END)
        word_count = len(re.findall(r'\b\w+\b', text))
        self.word_count_var.set(f"Words: {word_count}")

    def get_suggestions(self, word: str) -> list:
        """Get spelling suggestions for a word using multiple methods"""
        if not word or not self.autocorrect:
            return []

        suggestions = set()

        # Method 1: Get suggestions from autocorrect
        candidates = self.autocorrect._get_candidates(word.lower())
        suggestions.update(candidates)

        # Method 2: Use get_close_matches from difflib
        word_list = list(self.autocorrect.words)
        close_matches = get_close_matches(word.lower(), word_list, n=5, cutoff=0.6)
        suggestions.update(close_matches)

        # Method 3: Add words with similar length and common letters
        word_len = len(word)
        similar_words = [w for w in word_list
                         if abs(len(w) - word_len) <= 2
                         and any(c in w for c in word.lower())]
        suggestions.update(similar_words[:10])

        # Sort suggestions by frequency and similarity
        sorted_suggestions = sorted(
            suggestions,
            key=lambda x: (self.autocorrect.word_counts.get(x, 0),
                           -abs(len(x) - len(word))),
            reverse=True
        )

        # Return top 5 unique suggestions
        unique_suggestions = []
        seen = set()
        for s in sorted_suggestions:
            if s not in seen and s != word:
                seen.add(s)
                unique_suggestions.append(s)
                if len(unique_suggestions) == 5:
                    break

        return unique_suggestions

    def apply_suggestion(self, suggestion: str, start: str, end: str):
        """Apply the selected suggestion to the text"""
        self.text_widget.delete(start, end)
        self.text_widget.insert(start, suggestion)

    def setup_bindings(self):
        """Set up event bindings"""
        self.text_widget.bind("<Button-3>", self.show_context_menu)
        self.text_widget.bind("<KeyRelease>", self.on_text_change)

    def show_context_menu(self, event):
        """Show context menu with suggestions"""
        # Get the mouse click position in text coordinates
        click_index = self.text_widget.index(f"@{event.x},{event.y}")

        # Check if the click is on a misspelled word
        if "misspelled" in self.text_widget.tag_names(click_index):
            # Find the word boundaries
            start = self.text_widget.index(f"{click_index} wordstart")
            end = self.text_widget.index(f"{click_index} wordend")
            word = self.text_widget.get(start, end)

            # Get suggestions for the word
            suggestions = self.get_suggestions(word)

            # Clear and rebuild the suggestion menu
            self.suggestion_menu.delete(0, tk.END)
            if suggestions:
                for suggestion in suggestions:
                    self.suggestion_menu.add_command(
                        label=suggestion,
                        command=lambda s=suggestion, st=start, e=end:
                        self.apply_suggestion(s, st, e)
                    )
            else:
                self.suggestion_menu.add_command(
                    label="No suggestions available",
                    state=tk.DISABLED
                )

            # Show the context menu
            self.context_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def on_text_change(self, event):
        """Handle text change events"""
        self.status_var.set("Edited")

    def force_spell_check(self):
        """Force an immediate spell check"""
        self.last_check_time = 0
        self.check_spelling()

    def autocorrect_all(self):
        """Autocorrect all misspelled words in the text"""
        if not self.autocorrect:
            return

        text = self.text_widget.get("1.0", tk.END)
        corrected_text = self.autocorrect.correct_text(text)

        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", corrected_text)

    def new_file(self):
        """Create a new file"""
        self.text_widget.delete("1.0", tk.END)
        self.status_var.set("New file")

    def open_file(self):
        """Open a file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    self.text_widget.delete("1.0", tk.END)
                    self.text_widget.insert("1.0", content)
                self.status_var.set(f"Opened: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {str(e)}")

    def save_file(self):
        """Save the current file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                content = self.text_widget.get("1.0", tk.END)
                with open(file_path, 'w') as file:
                    file.write(content)
                self.status_var.set(f"Saved: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")


def main():
    # Create and run the application
    app = SpellCheckEditor()
    app.root.mainloop()


if __name__ == "__main__":
    main()
