"""CPU checks for score accounting and edit selection; no model dependencies."""
import unittest
import shared
from token_rdo import choose_edits, independent_change


class ScorePlanning(unittest.TestCase):
    def test_single_edit_score_delta_matches_full_objective(self):
        baseline={'seg':.0001254357,'pose':1.32737e-6}
        oldseg,newseg=.0001,.00015
        oldpose,newpose=1.1e-6,1.3e-6
        delta=independent_change(baseline,oldseg,oldpose,newseg,newpose)
        expected=shared.score(baseline['seg']+(newseg-oldseg)/600,
            baseline['pose']+(newpose-oldpose)/600,163872)-shared.score(
            baseline['seg'],baseline['pose'],163872)
        self.assertAlmostEqual(delta,expected,places=14)

    def test_rate_scale_matches_archive_score(self):
        a=shared.score(.0001,1e-6,163872)
        b=shared.score(.0001,1e-6,163872-1000)
        self.assertAlmostEqual(a-b,8000*shared.RATE_PER_BIT,places=14)

    def test_selection_orders_by_score_and_separates_interactions(self):
        rows=[dict(frame=0,position=0,estimated_score_change=-1.),
              dict(frame=0,position=1,estimated_score_change=-.9),
              dict(frame=1,position=0,estimated_score_change=-.8),
              dict(frame=0,position=20,estimated_score_change=-.7),
              dict(frame=2,position=0,estimated_score_change=.1)]
        chosen=choose_edits(rows,4)
        self.assertEqual([(r['frame'],r['position']) for r in chosen],[(0,0),(1,0),(0,20)])
        self.assertEqual(len(choose_edits(rows,1)),1)


if __name__=='__main__':
    unittest.main()
