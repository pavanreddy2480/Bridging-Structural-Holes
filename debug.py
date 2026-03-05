import sys
import traceback

with open("debug_log.txt", "w") as f:
    f.write("Starting...\n")
    try:
        from run_pipeline import run_full_pipeline
        f.write("Imported pipeline.\n")
        
        # Override stdout/stderr to completely capture everything
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "a")
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            def flush(self):
                self.terminal.flush()
                self.log.flush()
                
        sys.stdout = Logger("debug_log.txt")
        sys.stderr = sys.stdout
        
        print("Running pipeline...")
        run_full_pipeline(sample_size=20, epochs=1)
        print("SUCCESSFULLY FINISHED!")
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc(file=sys.stdout)
