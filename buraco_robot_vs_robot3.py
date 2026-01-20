import pygame
import sys
import os
from typing import List, Tuple
from cards import Card, Suit, affiche_carte
from boardplayer import BoardPlayer
from boardgame import BoardGame
from player import Player, Robot, RobotEasy
import boardplayer as bp

# Configuration
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
CARD_WIDTH = 60
CARD_HEIGHT = 84
FPS = 60

# Couleurs
BACKGROUND_COLOR = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT_COLOR = (255, 255, 0)

class BuracoRobotGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Buraco - Robot vs Robot (Mode Spectateur)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.big_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        self.load_assets()
        
        # Initialisation de deux robots
        self.p1 = RobotEasy("Robot 1", [], bp.BoardPlayer(), 0)
        self.p2 = RobotEasy("Robot 2", [], bp.BoardPlayer(), 0)
        self.game = BoardGame([self.p1, self.p2], is_open=True)
        
        self.curr_idx = 0
        self.message = "Match Robot vs Robot - Mode Spectateur"
        self.game_over = False
        
        # Timer pour cadencer le jeu (1 seconde entre chaque tour)
        self.action_timer = pygame.time.get_ticks() + 1000

    def load_assets(self):
        self.card_images = {}
        suits = ["hearts", "diamonds", "clubs", "spades"]
        for suit in suits:
            for val in range(1, 14):
                if val == 1: name = "ace"
                elif val == 11: name = "jack"
                elif val == 12: name = "queen"
                elif val == 13: name = "king"
                else: name = str(val)
                
                filename = f"{name}_of_{suit}.png"
                try:
                    img = pygame.image.load(filename)
                    self.card_images[(suit, val)] = pygame.transform.scale(img, (CARD_WIDTH, CARD_HEIGHT))
                except:
                    img = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
                    img.fill(WHITE)
                    self.card_images[(suit, val)] = img
        
        try:
            self.card_images["joker"] = pygame.transform.scale(pygame.image.load("joker.png"), (CARD_WIDTH, CARD_HEIGHT))
            self.back_image = pygame.transform.scale(pygame.image.load("back_side.png"), (CARD_WIDTH, CARD_HEIGHT))
        except:
            self.card_images["joker"] = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
            self.back_image = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
            self.back_image.fill((0, 0, 255))

    def get_card_image(self, card):
        if card.suit == Suit.JOKER:
            return self.card_images.get("joker")
        suit_map = {Suit.HEARTS: "hearts", Suit.DIAMONDS: "diamonds", Suit.CLUBS: "clubs", Suit.SPADES: "spades"}
        return self.card_images.get((suit_map.get(card.suit), card.value), self.back_image)

    def get_hand_card_pos(self, index, is_p2=False):
        margin_x = 10
        overlap_x = 20
        side_width = (SCREEN_WIDTH - 120) // 2 # Zone joueur élargie
        cards_per_row = (side_width - margin_x * 2 - CARD_WIDTH) // overlap_x + 1
        
        row = index // cards_per_row
        col = index % cards_per_row
        
        if is_p2:
            # Robot 2 en haut à droite
            base_x = SCREEN_WIDTH - side_width + margin_x
            base_y = 50
            x = base_x + col * overlap_x
            y = base_y + row * (CARD_HEIGHT + 10)
        else:
            # Robot 1 en bas à gauche
            base_x = margin_x
            base_y = SCREEN_HEIGHT - CARD_HEIGHT - 20
            x = base_x + col * overlap_x
            y = base_y - row * (CARD_HEIGHT + 10)
            
        return x, y

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Nouvelles dimensions des zones
        side_width = (SCREEN_WIDTH - 120) // 2
        mid_width = 120
        mid_x = side_width
        right_x = side_width + mid_width

        # Trier les cartes pour l'affichage
        suit_order = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3, Suit.JOKER: 4}
        self.p1.cards.sort(key=lambda c: (suit_order.get(c.suit, 5), c.value))
        self.p2.cards.sort(key=lambda c: (suit_order.get(c.suit, 5), c.value))
        
        # Séparateurs
        pygame.draw.line(self.screen, WHITE, (mid_x, 0), (mid_x, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (right_x, 0), (right_x, SCREEN_HEIGHT), 2)
        
        # Scores
        self.screen.blit(self.big_font.render(f"{self.p1.name}: {self.p1.score}", True, WHITE), (20, 20))
        self.screen.blit(self.big_font.render(f"{self.p2.name}: {self.p2.score}", True, WHITE), (right_x + 20, 20))
        
        # Message (centré dans la bande du milieu ou en bas)
        msg_surface = self.font.render(self.message, True, HIGHLIGHT_COLOR)
        self.screen.blit(msg_surface, (mid_x + 10, SCREEN_HEIGHT - 40))

        # --- Zone Milieu (Pioche et Poubelle) ---
        if not self.game.deck_empty():
            self.screen.blit(self.back_image, (mid_x + 30, 50))
            self.screen.blit(self.font.render("Deck", True, WHITE), (mid_x + 30, 30))
        
        self.screen.blit(self.font.render("Discard", True, WHITE), (mid_x + 30, 150))
        for i, card in enumerate(self.game.discard_pile):
            img = self.get_card_image(card)
            self.screen.blit(img, (mid_x + 30, 180 + i * 18))

        # --- Zone Robot 1 (Gauche) ---
        if self.p1.first_game and len(self.game.pots) > 0:
            pot_x = mid_x - CARD_WIDTH - 10
            self.screen.blit(self.font.render("Pot", True, WHITE), (pot_x, 5))
            self.screen.blit(self.back_image, (pot_x, 30))

        for i, card in enumerate(self.p1.cards):
            img = self.get_card_image(card)
            x, y = self.get_hand_card_pos(i, is_p2=False)
            self.screen.blit(img, (x, y))

        for i, game_list in enumerate(self.p1.board.cardgames):
            meld_cards = game_list[0]
            for j, card in enumerate(meld_cards):
                img = self.get_card_image(card)
                self.screen.blit(img, (10 + i * 65, 150 + j * 15))

        # --- Zone Robot 2 (Droite) ---
        if self.p2.first_game and len(self.game.pots) > 0:
            pot_x = SCREEN_WIDTH - CARD_WIDTH - 10
            pot_y = SCREEN_HEIGHT - CARD_HEIGHT - 50
            self.screen.blit(self.font.render("Pot", True, WHITE), (pot_x, pot_y - 25))
            self.screen.blit(self.back_image, (pot_x, pot_y))

        for i, card in enumerate(self.p2.cards):
            img = self.get_card_image(card)
            x, y = self.get_hand_card_pos(i, is_p2=True)
            self.screen.blit(img, (x, y))
            
        for i, game_list in enumerate(self.p2.board.cardgames):
            meld_cards = game_list[0]
            for j, card in enumerate(meld_cards):
                img = self.get_card_image(card)
                self.screen.blit(img, (right_x + 10 + i * 65, 250 + j * 15))

        pygame.display.flip()

    def update_game_logic(self):
        if self.game_over: return
        
        now = pygame.time.get_ticks()
        if now > self.action_timer:
            self.action_timer = now + 500 # 0.5 seconde entre chaque tour pour plus de vitesse
            
            curr_p = self.game.players[self.curr_idx]
            
            try:
                from main import robot_turn
                
                # Sauvegarde de l'état pour déduire l'action
                old_hand_size = len(curr_p.cards)
                old_board_games_count = len(curr_p.board.cardgames)
                old_discard_pile_size = len(self.game.discard_pile)
                
                should_stop = robot_turn(curr_p, self.game, self.curr_idx)
                
                # Déduction de l'action pour l'affichage
                action_msg = ""
                if len(self.game.discard_pile) > old_discard_pile_size:
                    action_msg = "Défausse"
                
                if len(curr_p.board.cardgames) > old_board_games_count:
                    action_msg = "Pose un nouveau jeu"
                elif any(len(g[0]) > 0 for g in curr_p.board.cardgames): # Simplification
                    action_msg = "Complète un jeu"
                
                if old_discard_pile_size > 0 and len(self.game.discard_pile) == 0:
                    action_msg = "Prend la poubelle"
                elif len(curr_p.cards) > old_hand_size:
                    action_msg = "Pioche"

                if not should_stop:
                    self.message = f"{curr_p.name} : {action_msg}" if action_msg else f"{curr_p.name} a joué."
                
                if should_stop:
                    self.game_over = True
                    self.message = f"PARTIE TERMINÉE ! {curr_p.name} a gagné."
                    return
                
                # Vérification du deck
                if self.game.deck_empty():
                    if not self.game.update_deck():
                        self.game_over = True
                        self.message = "Jeu terminé : Deck vide et plus de pots."
                        return
                
                # Passage au joueur suivant
                self.curr_idx = self.game.next_player_index(self.curr_idx)
                
            except Exception as e:
                self.message = f"Erreur : {str(e)}"
                print(f"Erreur lors du tour du robot : {e}")

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.update_game_logic()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = BuracoRobotGUI()
    gui.run()
