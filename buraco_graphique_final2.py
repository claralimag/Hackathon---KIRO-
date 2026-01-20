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

class BuracoGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Buraco - Humain vs Robot")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.big_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        self.load_assets()
        
        # Initialisation du jeu comme dans main.py
        self.p1 = Player("Alice", [], bp.BoardPlayer(), 0)
        self.p2 = RobotEasy("Bot", [], bp.BoardPlayer(), 0)
        self.game = BoardGame([self.p1, self.p2], is_open=True)
        
        self.curr_idx = 0
        
        self.selected_cards = []
        self.turn_phase = "draw" # "draw", "action", "discard"
        self.message = "C'est votre tour ! Piochez ou prenez la poubelle."
        self.game_over = False
        self.robot_timer = 0

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

    def get_hand_card_pos(self, index, is_robot=False):
        """Calcule la position d'une carte dans la main avec superposition horizontale."""
        margin_x = 10
        overlap_x = 20
        side_width = (SCREEN_WIDTH - 120) // 2
        cards_per_row = (side_width - margin_x * 2 - CARD_WIDTH) // overlap_x + 1
        
        row = index // cards_per_row
        col = index % cards_per_row
        
        if is_robot:
            # Main du robot en haut à droite
            base_x = SCREEN_WIDTH - side_width + margin_x
            base_y = 50
            x = base_x + col * overlap_x
            y = base_y + row * (CARD_HEIGHT + 10)
        else:
            # Main du joueur en bas à gauche
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

        # Trier les cartes du joueur par couleur puis valeur
        suit_order = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3, Suit.JOKER: 4}
        self.p1.cards.sort(key=lambda c: (suit_order.get(c.suit, 5), c.value))
        
        # Séparateurs
        pygame.draw.line(self.screen, WHITE, (mid_x, 0), (mid_x, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (right_x, 0), (right_x, SCREEN_HEIGHT), 2)
        
        # Scores
        self.screen.blit(self.big_font.render(f"your score: {self.p1.score}", True, WHITE), (20, 20))
        self.screen.blit(self.big_font.render(f"robot score: {self.p2.score}", True, WHITE), (right_x + 20, 20))
        
        # Message
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

        # --- Zone Joueur (Gauche) ---
        # Pot du joueur (face cachée) - Déplacé en haut à droite de sa zone, à côté du score
        if self.p1.first_game and len(self.game.pots) > 0:
            pot_x = mid_x - CARD_WIDTH - 10
            self.screen.blit(self.font.render("Pot", True, WHITE), (pot_x, 5))
            self.screen.blit(self.back_image, (pot_x, 30))

        # Main du joueur
        for i, card in enumerate(self.p1.cards):
            img = self.get_card_image(card)
            x, y = self.get_hand_card_pos(i, is_robot=False)
            rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
            if card in self.selected_cards:
                pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect.inflate(4, 4), 2)
            self.screen.blit(img, rect)

        # Melds du joueur
        for i, game_list in enumerate(self.p1.board.cardgames):
            meld_cards = game_list[0]
            for j, card in enumerate(meld_cards):
                img = self.get_card_image(card)
                self.screen.blit(img, (10 + i * 65, 150 + j * 15))

        # --- Zone Robot (Droite) ---
        # Pot du robot (face cachée)
        if self.p2.first_game and len(self.game.pots) > 0:
            pot_x = SCREEN_WIDTH - CARD_WIDTH - 10
            pot_y = SCREEN_HEIGHT - CARD_HEIGHT - 50
            self.screen.blit(self.font.render("Pot", True, WHITE), (pot_x, pot_y - 25))
            self.screen.blit(self.back_image, (pot_x, pot_y))

        # Main du robot (disposée comme l'humain)
        for i in range(len(self.p2.cards)):
            x, y = self.get_hand_card_pos(i, is_robot=True)
            self.screen.blit(self.back_image, (x, y))
            
        # Melds du robot
        for i, game_list in enumerate(self.p2.board.cardgames):
            meld_cards = game_list[0]
            for j, card in enumerate(meld_cards):
                img = self.get_card_image(card)
                self.screen.blit(img, (right_x + 10 + i * 65, 250 + j * 15))

        pygame.display.flip()

    def handle_click(self, pos):
        if self.game_over or self.curr_idx != 0: return
        side_width = (SCREEN_WIDTH - 120) // 2
        mid_x = side_width
        
        if self.turn_phase == "draw":
            if pygame.Rect(mid_x + 30, 50, CARD_WIDTH, CARD_HEIGHT).collidepoint(pos):
                card = self.game.draw_from_deck()
                self.p1.add_card(card)
                self.turn_phase = "action"
                self.message = "Action : Sélectionnez des cartes pour poser/étendre ou cliquez sur la poubelle pour défausser."
                return
            if self.game.discard_pile:
                discard_rect = pygame.Rect(mid_x + 30, 180, CARD_WIDTH, len(self.game.discard_pile) * 18 + CARD_HEIGHT)
                if discard_rect.collidepoint(pos):
                    cards = self.game.draw_from_discard()
                    self.p1.add_cards(cards)
                    self.turn_phase = "action"
                    self.message = "Action : Sélectionnez des cartes pour poser/étendre ou cliquez sur la poubelle pour défausser."
                    return

        if self.turn_phase == "action":
            for i in range(len(self.p1.cards) - 1, -1, -1):
                card = self.p1.cards[i]
                x, y = self.get_hand_card_pos(i, is_robot=False)
                if pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT).collidepoint(pos):
                    if card in self.selected_cards: self.selected_cards.remove(card)
                    else: self.selected_cards.append(card)
                    return

            # 1. Tester d'abord si on clique sur un jeu existant pour l'étendre
            for i, game_list in enumerate(self.p1.board.cardgames):
                # Zone de clic élargie pour couvrir toute la colonne du jeu existant
                meld_cards_count = len(game_list[0])
                meld_height = 15 * (meld_cards_count - 1) + CARD_HEIGHT
                meld_rect = pygame.Rect(10 + i * 65, 150, CARD_WIDTH, meld_height)
                if meld_rect.collidepoint(pos) and self.selected_cards:
                    self.try_play_cards(i)
                    return

            # 2. Tester ensuite si on clique sur le plateau pour créer un nouveau jeu
            player_board_rect = pygame.Rect(0, 150, side_width, 350)
            if player_board_rect.collidepoint(pos) and self.selected_cards:
                self.try_play_cards(-1)
                return

            if pygame.Rect(mid_x + 30, 180, CARD_WIDTH, 350).collidepoint(pos):
                if len(self.selected_cards) == 1:
                    card_to_discard = self.selected_cards[0]
                    self.p1.update_cards([card_to_discard])
                    self.game.add_to_discard(card_to_discard)
                    self.selected_cards = []
                    
                    # Vérifier si le joueur a fini sa main après défausse
                    if len(self.p1.cards) == 0 and not self.p1.first_game:
                        if self.game.can_end(0):
                            self.p1.score += 100
                            self.game_over = True
                            self.message = "Félicitations ! Vous avez gagné."
                            return

                    self.end_human_turn()
                elif not self.selected_cards:
                    self.turn_phase = "discard"
                    self.message = "Défausse : Cliquez sur une carte de votre main pour terminer le tour."
                else:
                    self.message = "Erreur : Sélectionnez une seule carte pour défausser."
                return

        if self.turn_phase == "discard":
            for i in range(len(self.p1.cards) - 1, -1, -1):
                card = self.p1.cards[i]
                x, y = self.get_hand_card_pos(i, is_robot=False)
                if pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT).collidepoint(pos):
                    self.p1.update_cards([card])
                    self.game.add_to_discard(card)
                    
                    # Vérifier si le joueur a fini sa main après défausse
                    if len(self.p1.cards) == 0 and not self.p1.first_game:
                        if self.game.can_end(0):
                            self.p1.score += 100
                            self.game_over = True
                            self.message = "Félicitations ! Vous avez gagné."
                            return

                    self.end_human_turn()
                    return

    def try_play_cards(self, game_index):
        if not self.selected_cards:
            return

        cards_to_play = self.selected_cards[:]
        # Sauvegarde pour annulation en cas d'échec
        original_hand = self.p1.cards[:]
        original_score = self.p1.score
        original_first_game = self.p1.first_game

        try:
            # 1. Retirer les cartes de la main (comme dans main.py ligne 128)
            hand_nonempty = self.p1.update_cards(cards_to_play)

            # 2. Gérer le cas de la main vide (pot ou fin) AVANT de poser (comme dans main.py ligne 131)
            if not hand_nonempty:
                if self.p1.first_game:
                    try:
                        new_pot = self.game.take_pot()
                        self.p1.add_cards(new_pot)
                        self.p1.score += 100
                        self.p1.first_game = False
                        self.message = "Vous avez pris le pot ! (+100 pts)"
                    except RuntimeError:
                        if self.game.can_end(0):
                            self.p1.score += 100
                            self.game_over = True
                            self.message = "Félicitations ! Vous avez gagné."
                            return
                        else:
                            # Si on ne peut pas finir, on doit annuler le coup car la main est vide
                            raise ValueError("Vous ne pouvez pas finir le jeu maintenant.")
                else:
                    if self.game.can_end(0):
                        self.p1.score += 100
                        self.game_over = True
                        self.message = "Félicitations ! Vous avez gagné."
                        return
                    else:
                        raise ValueError("Vous ne pouvez pas finir le jeu maintenant.")

            # 3. Tenter de poser les cartes sur le plateau (comme dans main.py ligne 153)
            # Note: play_cards dans votre player.py appelle board.add_to_board
            points = self.p1.play_cards(cards_to_play, game_index)
            
            if points > 0:
                self.message = f"Coup réussi ! (+{points} pts). Continuez ou défaussez."
            else:
                # Si points == 0, le coup est invalide selon boardplayer.py
                raise ValueError("Combinaison invalide.")

        except (ValueError, IndexError, Exception) as e:
            # En cas d'erreur, on restaure l'état précédent (Annulation)
            self.p1.cards = original_hand
            self.p1.score = original_score
            self.p1.first_game = original_first_game
            self.message = f"Action impossible : {str(e)}"
                
        self.selected_cards = []

    def end_human_turn(self):
        if self.game_over: return
        if self.game.deck_empty():
            if not self.game.update_deck():
                self.game_over = True
                self.message = "Jeu terminé : Deck vide."
                return
        self.curr_idx = self.game.next_player_index(self.curr_idx)
        self.robot_timer = pygame.time.get_ticks() + 1000
        self.message = "Le robot réfléchit..."

    def robot_turn_logic(self):
        if self.robot_timer > 0 and pygame.time.get_ticks() > self.robot_timer:
            self.robot_timer = 0
            try:
                from main import robot_turn
                should_stop = robot_turn(self.p2, self.game, self.curr_idx)
                if should_stop:
                    self.game_over = True
                    self.message = "Le robot a gagné."
                    return
                if self.game.deck_empty():
                    if not self.game.update_deck():
                        self.game_over = True
                        self.message = "Jeu terminé : Deck vide."
                        return
                self.curr_idx = self.game.next_player_index(self.curr_idx)
                self.turn_phase = "draw"
                self.message = "C'est votre tour ! Piochez ou prenez la poubelle."
            except Exception as e:
                self.message = f"Erreur Robot : {str(e)}"
                self.curr_idx = 0
                self.turn_phase = "draw"

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
            
            if not self.game_over and self.curr_idx != 0:
                self.robot_turn_logic()
                
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = BuracoGUI()
    gui.run()
