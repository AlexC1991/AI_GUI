"""
Admin Console for IronGate
"""
import os
import threading
import time
import sys
from colorama import Fore, Style
from .config import VERSION
from .database import client_db, save_database
from .client_gen import generate_client_package
from .security import reset_ip, ban_ip, kick_client, trust_ip

def start_admin_console(tunnel_url, db_callback, gen_callback):
    """Start admin console thread"""
    t = threading.Thread(target=admin_loop, args=(tunnel_url, db_callback, gen_callback), daemon=True)
    t.start()

def admin_loop(tunnel_url, db_callback, gen_callback):
    time.sleep(2)
    print(f"\n{Fore.GREEN}{Style.BRIGHT}=== ADMIN CONSOLE ===")
    print(f"{Fore.WHITE}Type 'help' for commands.")
    
    while True:
        try:
            raw = input(f"{Fore.BLUE}Admin> {Fore.WHITE}").strip()
            if not raw: continue
            
            parts = raw.split()
            cmd = parts[0].lower()
            args = parts[1:]
            
            if cmd in ["help", "?"]:
                print(f"{Fore.YELLOW}Available Commands:")
                print(f"  {Fore.CYAN}users, list{Fore.WHITE}   : List all known clients and active sessions")
                print(f"  {Fore.CYAN}gen, generate{Fore.WHITE} : Generate a new client (Usage: gen \"Name\")")
                print(f"  {Fore.CYAN}reset{Fore.WHITE}         : Reset IP ban/auth (Usage: reset <IP>)")
                print(f"  {Fore.CYAN}ban{Fore.WHITE}           : Ban an IP manually (Usage: ban <IP>)")
                print(f"  {Fore.CYAN}kick{Fore.WHITE}          : Kick a client (Usage: kick <ID>)")
                print(f"  {Fore.CYAN}cls, clear{Fore.WHITE}    : Clear console")
                print(f"  {Fore.CYAN}q, exit{Fore.WHITE}       : Quit IronGate Host")
            
            elif cmd in ["list", "users"]:
                clients = client_db.get("clients", {})
                print(f"\n{Fore.WHITE}--- Known Clients ---")
                for cid, c in clients.items():
                    status_color = Fore.GREEN if c['status'] == 'ACTIVE' else Fore.YELLOW
                    print(f"{Fore.CYAN}{cid}: {Fore.WHITE}{c['name']} [{status_color}{c['status']}{Fore.WHITE}]")
                
                print(f"\n{Fore.WHITE}--- Active Sessions ---")
                # We need to peek at security.authenticated_ips or similar? 
                # For now just database view is okay, but real-time sessions would be better.
                # Assuming client_db tracks status reasonably well.

            elif cmd in ["generate", "gen"]:
                if not args:
                    print(f"{Fore.RED}Usage: generate \"Client Name\"")
                    continue
                name = " ".join(args).replace('"', '')
                print(f"Generating for {name}...")
                success, res = gen_callback(name, tunnel_url, db_callback)
                if success:
                    print(f"{Fore.GREEN}Done!")
                    print(f"{Fore.WHITE}  Activation Code : {Fore.YELLOW}{res['install_code']}")
                    print(f"{Fore.WHITE}  Client ID       : {res['client_id']}")
                    print(f"{Fore.WHITE}  Executable      : {res['exe_name']}")
                    print(f"{Fore.WHITE}  Package folder  : {res.get('package_path', '')}")
                    if res.get('zip_name'):
                        print(f"{Fore.WHITE}  ZIP file        : {Fore.CYAN}{res['zip_name']}")

            elif cmd in ["reset"]:
                if not args:
                    print(f"{Fore.RED}Usage: reset <IP>")
                    continue
                target = args[0]
                if reset_ip(target):
                    print(f"{Fore.GREEN}Reset successful for {target}")
                else:
                    print(f"{Fore.YELLOW}Target not found in ban/fail list.")
            
            elif cmd in ["ban"]:
                if not args:
                    print(f"{Fore.RED}Usage: ban <IP>")
                    continue
                target = args[0]
                if ban_ip(target):
                    print(f"{Fore.RED}Banned IP: {target}")
                else:
                    print(f"{Fore.YELLOW}Error banning IP.")

            elif cmd in ["kick"]:
                if not args:
                    print(f"{Fore.RED}Usage: kick <Client ID>")
                    continue
                target = args[0]
                if kick_client(target):
                    print(f"{Fore.GREEN}Kicked client: {target}")
                else:
                    print(f"{Fore.YELLOW}Client not found or not active.")

            elif cmd in ["trust", "allow"]:
                if not args:
                    print(f"{Fore.RED}Usage: trust <IP>")
                    continue
                target = args[0]
                if trust_ip(target):
                    print(f"{Fore.GREEN}Trusted IP for Host Access: {target}")

            elif cmd in ["cls", "clear"]:
                import os
                os.system("cls" if os.name == "nt" else "clear")
                print(f"{Fore.GREEN}{Style.BRIGHT}=== IRON TUNNEL HOST (Admin) ===")

            elif cmd in ["exit", "q", "quit"]:
                print(f"{Fore.RED}Shutting down...")
                import os
                os._exit(0)
            
        except Exception as e:
            print(f"Error: {e}")

